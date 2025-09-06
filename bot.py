import streamlit as st
import os
import json
import pickle
import requests
from pathlib import Path
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# Page config
st.set_page_config(
    page_title="RAG Chatbot", 
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

@st.cache_resource
def load_sentence_transformer():
    """Load the sentence transformer model for embeddings"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading sentence transformer: {e}")
        return None

@st.cache_resource
def init_groq():
    """Initialize Groq client"""
    # Try to get from secrets/environment first, then fallback to hardcoded (for testing only)
    api_key = (os.getenv("GROQ_API_KEY") or 
               st.secrets.get("GROQ_API_KEY", "") or 
               "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")
    
    if api_key:
        return Groq(api_key=api_key)
    return None

@st.cache_data
def load_chunks():
    """Load chunks from files"""
    chunks = []
    chunk_files = []
    
    # Look for chunk files in current directory and chunks folder
    for pattern in ["chunk*.txt", "chunks/chunk*.txt", "chunks/*.txt"]:
        chunk_files.extend(Path(".").glob(pattern))
    
    # Also try to load from chunk_index.txt if it exists
    if Path("chunk_index.txt").exists():
        try:
            with open("chunk_index.txt", "r", encoding="utf-8") as f:
                content = f.read()
                # Split by double newlines or numbered sections
                if "1." in content and "2." in content:
                    chunks = content.split("\n\n")
                else:
                    chunks = [content]
        except Exception as e:
            st.error(f"Error reading chunk_index.txt: {e}")
    
    # Load individual chunk files
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_content = f.read().strip()
                if chunk_content:
                    chunks.append(chunk_content)
        except Exception as e:
            st.error(f"Error reading {chunk_file}: {e}")
    
    # If no chunks found, create some sample data
    if not chunks:
        chunks = [
            "Welcome to the chatbot. This is sample content chunk 1.",
            "This chatbot uses RAG (Retrieval Augmented Generation) to provide contextual responses.",
            "You can upload your own documents to create custom knowledge base chunks."
        ]
    
    return chunks

@st.cache_resource
def create_vector_store(_chunks, _model):
    """Create FAISS vector store from chunks"""
    if not _chunks or not _model:
        return None, []
    
    try:
        # Generate embeddings for all chunks
        embeddings = _model.encode(_chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return index, embeddings
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, []

def search_similar_chunks(query, index, chunks, model, k=3):
    """Search for similar chunks using vector similarity"""
    if not index or not model or not chunks:
        return []
    
    try:
        # Generate embedding for query
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(chunks) and score > 0.1:  # Minimum similarity threshold
                results.append({
                    'content': chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    except Exception as e:
        st.error(f"Error searching chunks: {e}")
        return []

def get_groq_response(message, context=""):
    """Get response from Groq API"""
    client = init_groq()
    if not client:
        return "❌ Please set your Groq API key in Streamlit secrets."
    
    try:
        # Create system prompt with context
        system_prompt = """You are a helpful AI assistant. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so and provide a general response."""
        
        if context:
            system_prompt += f"\n\nContext information:\n{context}"
        
        response = client.chat.completions.create(
            model="llama2-70b-4096",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error getting response: {str(e)}"

def get_openai_response(message, context=""):
    """Get response from OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return "❌ Please set your OpenAI API key in Streamlit secrets."
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = "You are a helpful AI assistant. Use the provided context to answer questions accurately."
        if context:
            system_prompt += f"\n\nContext: {context}"
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Error getting OpenAI response: {str(e)}"

# Load resources
model = load_sentence_transformer()
chunks = load_chunks()

# Create vector store
if model and chunks and st.session_state.vector_store is None:
    with st.spinner("Creating vector store..."):
        st.session_state.vector_store, st.session_state.embeddings = create_vector_store(chunks, model)
        st.session_state.chunks = chunks

# Main UI
st.title("🤖 RAG Chatbot")
st.write("Ask questions and get contextual answers from your knowledge base!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Selection
    api_choice = st.selectbox(
        "Choose AI API:",
        ["Groq (Llama2)", "OpenAI (GPT-3.5)"],
        help="Select your preferred AI API"
    )
    
    # Chunk info
    st.header("📚 Knowledge Base")
    st.write(f"**Loaded chunks:** {len(chunks)}")
    st.write(f"**Vector store:** {'✅ Ready' if st.session_state.vector_store is not None else '❌ Not ready'}")
    
    # Show sample chunks
    if st.checkbox("Show sample chunks"):
        for i, chunk in enumerate(chunks[:3]):
            with st.expander(f"Chunk {i+1}"):
                st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    st.header("🔧 Setup")
    st.markdown("""
    **Required API Keys:**
    
    Add to Streamlit secrets:
    ```toml
    GROQ_API_KEY = "your-groq-key"
    OPENAI_API_KEY = "your-openai-key"
    ```
    
    **Get API Keys:**
    - [Groq Console](https://console.groq.com) (Free)
    - [OpenAI Platform](https://platform.openai.com) (Paid)
    """)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and st.checkbox(f"Show context for message {len(st.session_state.messages)}", key=f"context_{len(st.session_state.messages)}"):
            with st.expander("📄 Context used"):
                for i, ctx in enumerate(message["context"]):
                    st.write(f"**Chunk {i+1}** (Score: {ctx['score']:.3f})")
                    st.write(ctx["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # Search for relevant chunks
            relevant_chunks = []
            context = ""
            
            if st.session_state.vector_store is not None and model:
                relevant_chunks = search_similar_chunks(
                    prompt, 
                    st.session_state.vector_store, 
                    st.session_state.chunks, 
                    model, 
                    k=3
                )
                
                if relevant_chunks:
                    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
                    st.success(f"Found {len(relevant_chunks)} relevant chunks")
                else:
                    st.info("No highly relevant chunks found. Providing general response.")
        
        with st.spinner("Generating response..."):
            # Get response from chosen API
            if api_choice == "Groq (Llama2)":
                response = get_groq_response(prompt, context)
            else:
                response = get_openai_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response with context
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "context": relevant_chunks
    })

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
