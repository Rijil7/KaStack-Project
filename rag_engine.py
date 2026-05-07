import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from data_parser import load_conversations
import json

CHROMA_DATA_PATH = "chroma_db"

def chunk_messages(messages, window_size=5):
    chunks = []
    for i in range(0, len(messages), window_size):
        chunk = messages[i:i+window_size]
        text = "\n".join([f"{m['speaker']}: {m['text']}" for m in chunk])
        chunks.append(text)
    return chunks

def detect_topic_boundaries(messages, window_size=10, epsilon=0.5):
    print("Loading embedding model for topic boundary detection...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create text windows
    windows = []
    for i in range(0, len(messages), window_size):
        chunk = messages[i:i+window_size]
        text = " ".join([m['text'] for m in chunk])
        windows.append(text)
        
    print(f"Computing embeddings for {len(windows)} windows...")
    embeddings = model.encode(windows, show_progress_bar=True)
    
    boundaries = [0]
    for i in range(len(embeddings) - 1):
        # Cosine similarity
        v1 = embeddings[i]
        v2 = embeddings[i+1]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if sim < epsilon:
            boundaries.append((i+1) * window_size)
    boundaries.append(len(messages))
    
    topics = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        topics.append(messages[start:end])
    return topics

def summarize_text(text, summarizer, max_length=50):
    # Truncate text to avoid model max length errors
    truncated = text[:1024]
    try:
        summary = summarizer(truncated, max_length=max_length, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Fallback to extractive if generation fails
        return truncated[:200] + "..."

def build_vector_store(messages_limit=200):
    messages = load_conversations('conversations.csv')
    if messages_limit:
        messages = messages[:messages_limit]
        print(f"Limiting to {messages_limit} messages for processing time.")
        
    print("Detecting topic boundaries...")
    topic_segments = detect_topic_boundaries(messages, window_size=10, epsilon=0.5)
    print(f"Detected {len(topic_segments)} topics.")
    
    # 100-message segments
    hundred_segments = []
    for i in range(0, len(messages), 100):
        hundred_segments.append(messages[i:i+100])
        
    print("Summarizing topics...")
    topic_summaries = []
    for seg in tqdm(topic_segments):
        text = "\n".join([f"{m['speaker']}: {m['text']}" for m in seg])
        topic_summaries.append(text[:200] + "...")
        
    print("Summarizing 100-message segments...")
    hundred_summaries = []
    for seg in tqdm(hundred_segments):
        text = "\n".join([f"{m['speaker']}: {m['text']}" for m in seg])
        hundred_summaries.append(text[:200] + "...")
        
    print("Preparing Raw chunks...")
    raw_chunks = []
    for seg in topic_segments:
        raw_chunks.append("\n".join([f"{m['speaker']}: {m['text']}" for m in seg]))
        
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Topic Index
    topic_col = client.get_or_create_collection("topic_summaries", embedding_function=emb_func)
    # Check if exists to clear or just overwrite? Create new docs if collection is empty
    if topic_col.count() > 0:
        print("Warning: Collections already exist and have documents.")
        
    topic_col.add(
        documents=topic_summaries,
        metadatas=[{"type": "topic_summary"}] * len(topic_summaries),
        ids=[f"topic_{i}" for i in range(len(topic_summaries))]
    )
    
    # 100-msg Index
    hundred_col = client.get_or_create_collection("hundred_summaries", embedding_function=emb_func)
    hundred_col.add(
        documents=hundred_summaries,
        metadatas=[{"type": "hundred_summary"}] * len(hundred_summaries),
        ids=[f"hundred_{i}" for i in range(len(hundred_summaries))]
    )
    
    # Raw Index
    raw_col = client.get_or_create_collection("raw_chunks", embedding_function=emb_func)
    raw_col.add(
        documents=raw_chunks,
        metadatas=[{"type": "raw_chunk"}] * len(raw_chunks),
        ids=[f"raw_{i}" for i in range(len(raw_chunks))]
    )
    
    print("Vector store built successfully.")

def query_rag(user_question, k=3):
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    topic_col = client.get_collection("topic_summaries", embedding_function=emb_func)
    raw_col = client.get_collection("raw_chunks", embedding_function=emb_func)
    
    topic_res = topic_col.query(query_texts=[user_question], n_results=k)
    raw_res = raw_col.query(query_texts=[user_question], n_results=k)
    
    context_parts = []
    if topic_res['documents'] and len(topic_res['documents']) > 0:
        context_parts.extend(topic_res['documents'][0])
    if raw_res['documents'] and len(raw_res['documents']) > 0:
        context_parts.extend(raw_res['documents'][0])
        
    context = "\n\n---\n\n".join(context_parts)
    return context

if __name__ == '__main__':
    build_vector_store()
