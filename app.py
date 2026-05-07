import streamlit as st
import json
import os
from rag_engine import query_rag

def load_persona():
    if os.path.exists('user_persona.json'):
        with open('user_persona.json', 'r') as f:
            return json.load(f)
    return None

st.set_page_config(page_title="RAG Persona Chatbot", layout="wide")

st.title("User Persona & RAG Chatbot")

persona = load_persona()
if persona:
    with st.sidebar:
        st.header("Extracted User Persona")
        st.json(persona)
else:
    with st.sidebar:
        st.warning("user_persona.json not found. Please run persona_extractor.py first.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the user or their history..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Routing Logic
        p_q = ["what kind of person", "habits", "how do they talk", "who is", "persona", "traits"]
        if any(q in prompt.lower() for q in p_q) and persona:
            response = f"Based on the extracted persona, here is what I know about the user:\n\n"
            response += f"- **Habits:** {', '.join(persona.get('habits', []))}\n"
            response += f"- **Facts:** {', '.join(persona.get('personal_facts', []))}\n"
            response += f"- **Traits:** {', '.join(persona.get('personality_traits', []))}\n"
            response += f"- **Tone:** {persona.get('communication_style', {}).get('tone', 'unknown')}\n"
        else:
            # Query RAG
            try:
                context = query_rag(prompt, k=2)
                
                # Simple synthesis 
                response = f"**Answer based on conversation history:**\n\n*(Here is the retrieved context)*\n\n{context}\n\n"
                response += "*(Note: In a full deployment, a generative LLM would synthesize this context into a direct answer.)*"
            except Exception as e:
                response = f"Error querying RAG system: {e}\nPlease ensure `rag_engine.py` has been run to build the vector store."

        message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
