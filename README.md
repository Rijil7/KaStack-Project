# Persona RAG Chatbot

This project implements an end-to-end Chatbot, RAG system, and Persona Extraction pipeline based on chronological conversation data.

## Mathematical Approach for Topic Boundary Detection
The system calculates embeddings for a sliding window of messages using the `all-MiniLM-L6-v2` local model from `sentence-transformers`. It computes the cosine similarity between sequential windows:
$Similarity = \frac{A \cdot B}{||A|| ||B||}$
If the similarity drops below a tuned threshold ($\epsilon = 0.5$), a topic boundary is marked. The messages are split at these boundaries to group them into coherent topics.

## Hybrid RAG Retrieval System
The RAG system chunks the text into topic-based segments and fixed 100-message segments. These segments are summarized using a lightweight LLM (`sshleifer/distilbart-cnn-12-6` for fast local execution). The vector database (`ChromaDB`) holds three indices:
1. Topic summaries
2. 100-message summaries
3. Raw chunks

When queried, the engine fetches the top-K relevant topic summaries and raw chunks to provide rich context to the generation model.

## Persona Extraction Pipeline
The persona extraction logic uses `spaCy` to extract entity-based facts and habits from the user's messages via rule-based heuristics. A zero-shot classification pipeline (`facebook/bart-large-mnli`) assesses the overarching tone and personality traits. Basic statistics calculate communication style metrics like average word count and emoji usage. The structured JSON is saved to `user_persona.json`.

## Setup Instructions

1. Clone the repository and navigate to the directory.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```
3. Run the setup scripts to parse data, extract persona, and build the vector database:
   ```bash
   python persona_extractor.py
   python rag_engine.py
   ```
   *(Note: Processing large datasets locally may take time. The scripts have limits set for quick testing.)*
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Cloud Deployment

A `Dockerfile` is provided for easy deployment to cloud services like Render, Fly.io, or Hugging Face Spaces.

## Demo Links

- **Loom Video Demo**: 
- **Live Cloud URL**: https://kastack-project-fnf8kfufehyr6wrz72qwrv.streamlit.app/
