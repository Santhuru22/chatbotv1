import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# LangChain imports
try:
    from langchain.docstore.document import Document
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.vectorstores import FAISS
    from langchain_community.llms import Ollama
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    st.error(f"LangChain not available: {e}")
    st.info("Install with: pip install langchain langchain-community faiss-cpu streamlit")

class VislonaRAGChatbot:
    """
    RAG-based chatbot using Ollama for both embeddings and text generation
    """

    def __init__(self,
                 vector_store_path: str = "faiss_index_ollama",
                 embedding_model: str = "nomic-embed-text",
                 chat_model: str = "gemma3:1b",
                 ollama_base_url: str = "http://localhost:11434",
                 max_context_chunks: int = 3):
        """
        Initialize the RAG chatbot

        Args:
            vector_store_path: Path to saved FAISS vector store
            embedding_model: Ollama model for embeddings
            chat_model: Ollama model for chat responses
            ollama_base_url: Ollama server URL
            max_context_chunks: Maximum number of context chunks to use
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.ollama_base_url = ollama_base_url
        self.max_context_chunks = max_context_chunks

        self.vector_store = None
        self.embeddings = None
        self.llm = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize embeddings, LLM, and load vector store"""
        try:
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.ollama_base_url
            )

            # Initialize LLM
            self.llm = Ollama(
                model=self.chat_model,
                base_url=self.ollama_base_url,
                temperature=0.7
            )

            # Load vector store
            if Path(self.vector_store_path).exists():
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success("✅ Vector store loaded successfully!")
            else:
                st.error(f"❌ Vector store not found at: {self.vector_store_path}")
                st.info("Run the vector store creation script first!")

        except Exception as e:
            st.error(f"❌ Error initializing components: {e}")
            self.vector_store = None

    def get_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector store

        Args:
            query: User query

        Returns:
            List of relevant chunks with metadata
        """
        if not self.vector_store:
            return []

        try:
            # Search for relevant documents
            results = self.vector_store.similarity_search_with_score(
                query, k=self.max_context_chunks
            )

            context_chunks = []
            for doc, score in results:
                chunk = {
                    "content": doc.page_content,
                    "score": float(score),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                }
                context_chunks.append(chunk)

            return context_chunks

        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return []

    def format_context_for_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks into a prompt-friendly format

        Args:
            context_chunks: List of relevant chunks

        Returns:
            Formatted context string
        """
        if not context_chunks:
            return ""

        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"Context {i}:\n{chunk['content']}")

        return "\n\n".join(context_parts)

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using Ollama LLM with context

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Generated response
        """
        if not self.llm:
            return "❌ LLM not available. Please check Ollama setup."

        # Create prompt with context
        prompt = f"""You are Vislona AI Assistant, a helpful chatbot for the Vislona AI platform. Use the provided context to answer the user's question accurately and helpfully.

Context Information:
{context}

User Question: {query}

Instructions:
- Answer based on the provided context when relevant
- Be helpful, accurate, and concise
- If the context doesn't contain relevant information, provide a general helpful response
- Always maintain a friendly and professional tone
- If asked about Vislona features, refer to the context provided

Response:"""

        try:
            # Generate response using Ollama
            response = self.llm.invoke(prompt)
            return response

        except Exception as e:
            return f"❌ Error generating response: {e}"

    def chat(self, query: str) -> Dict[str, Any]:
        """
        Main chat function that combines retrieval and generation

        Args:
            query: User query

        Returns:
            Dictionary with response and metadata
        """
        # Get relevant context
        context_chunks = self.get_relevant_context(query)

        # Format context
        context = self.format_context_for_prompt(context_chunks)

        # Generate response
        response = self.generate_response(query, context)

        return {
            "response": response,
            "context_chunks": context_chunks,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }

def check_ollama_status():
    """Check if Ollama is running and what models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, [model['name'] for model in models]
        return False, []
    except:
        return False, []

def main():
    """
    Streamlit app for Vislona chatbot interface
    """
    st.set_page_config(
        page_title="Vislona AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Header
    st.title("🤖 Vislona AI Chatbot")
    st.markdown("*Powered by Ollama & Vector Search*")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check Ollama status
        ollama_running, available_models = check_ollama_status()

        if ollama_running:
            st.success("✅ Ollama is running")
            st.write(f"Available models: {', '.join(available_models)}")
        else:
            st.error("❌ Ollama not running")
            st.info("Start with: `ollama serve`")
            return

        # Model selection
        if available_models:
            # Embedding model selection
            embedding_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
            available_embedding = [m for m in available_models if any(em in m for em in embedding_models)]

            if available_embedding:
                selected_embedding = st.selectbox(
                    "Embedding Model",
                    available_embedding,
                    index=0
                )
            else:
                st.warning("No dedicated embedding models found")
                st.info("Install with: `ollama pull nomic-embed-text`")
                selected_embedding = st.selectbox(
                    "Use Chat Model for Embeddings",
                    available_models,
                    index=0
                )

            # Chat model selection
            selected_chat = st.selectbox(
                "Chat Model",
                available_models,
                index=0
            )

            # Other settings
            max_chunks = st.slider("Max Context Chunks", 1, 5, 3)
            vector_store_path = st.text_input(
                "Vector Store Path",
                value="faiss_index_ollama"
            )
        else:
            st.error("No models available")
            return

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("🔄 Initializing chatbot..."):
            st.session_state.chatbot = VislonaRAGChatbot(
                vector_store_path=vector_store_path,
                embedding_model=selected_embedding,
                chat_model=selected_chat,
                max_context_chunks=max_chunks
            )

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I'm the Vislona AI Assistant. I can help you with questions about the Vislona platform, AI model training, deployment, and more. How can I assist you today?"
        })

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about Vislona..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                if st.session_state.chatbot and st.session_state.chatbot.vector_store:
                    # Get chatbot response
                    chat_result = st.session_state.chatbot.chat(prompt)
                    response = chat_result["response"]
                    context_chunks = chat_result["context_chunks"]

                    # Display response
                    st.markdown(response)

                    # Show context sources in expander
                    if context_chunks:
                        with st.expander("📚 Sources Used", expanded=False):
                            for i, chunk in enumerate(context_chunks, 1):
                                st.write(f"**Source {i}** (Score: {chunk['score']:.3f}, Chunk: {chunk['chunk_id']})")
                                st.write(f"```\n{chunk['content'][:300]}{'...' if len(chunk['content']) > 300 else ''}\n```")

                else:
                    response = "❌ Sorry, I'm having trouble accessing the knowledge base. Please check if the vector store is properly set up."
                    st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar info
    with st.sidebar:
        st.header("💡 Tips")
        st.markdown("""
        **Ask me about:**
        - Vislona platform features
        - AI model training
        - Deployment processes
        - Pricing and plans
        - Team collaboration
        - File formats
        - Security features
        - Internship opportunities
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

def run_console_chatbot():
    """
    Console-based chatbot for command-line usage
    """
    print("🤖 Vislona AI Chatbot (Console Mode)")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'clear' to clear chat history")
    print("=" * 50)

    # Initialize chatbot
    chatbot = VislonaRAGChatbot()

    if not chatbot.vector_store:
        print("❌ Vector store not available. Run the vector store creation script first.")
        return

    chat_history = []

    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'clear':
                chat_history = []
                print("🗑️ Chat history cleared!")
                continue

            if not user_input:
                continue

            # Get chatbot response
            print("🤖 Vislona AI: ", end="", flush=True)
            chat_result = chatbot.chat(user_input)

            response = chat_result["response"]
            context_chunks = chat_result["context_chunks"]

            print(response)

            # Show sources if available
            if context_chunks:
                print(f"\n📚 Sources used: {len(context_chunks)} chunks")
                for i, chunk in enumerate(context_chunks, 1):
                    print(f"   • Chunk {chunk['chunk_id']} (Score: {chunk['score']:.3f})")

            # Add to chat history
            chat_history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": datetime.now().isoformat(),
                "sources": len(context_chunks)
            })

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def create_simple_web_interface():
    """
    Simple HTML-based chatbot interface
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vislona AI Chatbot</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .chat-container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(45deg, #4a6cf7, #667eea);
                color: white;
                padding: 20px;
                text-align: center;
            }
            .chat-messages {
                height: 500px;
                overflow-y: auto;
                padding: 20px;
                background: #f8f9fa;
            }
            .message {
                margin: 10px 0;
                padding: 15px;
                border-radius: 10px;
                max-width: 80%;
            }
            .user-message {
                background: #4a6cf7;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            .bot-message {
                background: white;
                border: 1px solid #e0e0e0;
                margin-right: auto;
            }
            .chat-input {
                display: flex;
                padding: 20px;
                background: white;
                border-top: 1px solid #e0e0e0;
            }
            .chat-input input {
                flex: 1;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 25px;
                outline: none;
                font-size: 16px;
            }
            .chat-input button {
                margin-left: 10px;
                padding: 15px 30px;
                background: #4a6cf7;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
            }
            .chat-input button:hover {
                background: #3b5ce6;
            }
            .sources {
                font-size: 12px;
                color: #666;
                margin-top: 10px;
                padding: 10px;
                background: #f0f0f0;
                border-radius: 5px;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 Vislona AI Chatbot</h1>
                <p>Powered by Ollama & Vector Search</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    <strong>Vislona AI:</strong> 👋 Hello! I'm the Vislona AI Assistant. I can help you with questions about the Vislona platform, AI model training, deployment, and more. How can I assist you today?
                </div>
            </div>
            
            <div class="loading" id="loading">
                <p>🤔 Thinking...</p>
            </div>
            
            <div class="chat-input">
                <input type="text" id="userInput" placeholder="Ask me anything about Vislona..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message
                addMessage(message, 'user');
                input.value = '';
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                
                // Send to backend (you'll need to implement this endpoint)
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({query: message})
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    addMessage(data.response, 'bot', data.context_chunks);
                })
                .catch(error => {
                    document.getElementById('loading').style.display = 'none';
                    addMessage('❌ Sorry, I encountered an error. Please try again.', 'bot');
                });
            }

            function addMessage(content, sender, sources = null) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                
                if (sender === 'user') {
                    messageDiv.innerHTML = `<strong>You:</strong> ${content}`;
                } else {
                    let sourcesHtml = '';
                    if (sources && sources.length > 0) {
                        sourcesHtml = `<div class="sources">📚 Sources: ${sources.length} chunks used</div>`;
                    }
                    messageDiv.innerHTML = `<strong>Vislona AI:</strong> ${content}${sourcesHtml}`;
                }
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

    # Save HTML file
    with open("vislona_chatbot.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("✅ Simple web interface created: vislona_chatbot.html")
    print("💡 For full functionality, you'll need to implement the /chat endpoint")

def create_flask_backend():
    """
    Create a Flask backend for the web interface
    """
    flask_code = '''
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Initialize chatbot (make sure vector store exists)
chatbot = VislonaRAGChatbot()

@app.route('/')
def index():
    # Serve the HTML interface
    with open('vislona_chatbot.html', 'r') as f:
        return f.read()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get chatbot response
        result = chatbot.chat(query)
        
        return jsonify({
            'response': result['response'],
            'context_chunks': result['context_chunks'],
            'timestamp': result['timestamp']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'chatbot_ready': chatbot.vector_store is not None})

if __name__ == '__main__':
    print("🚀 Starting Vislona Chatbot Server...")
    print("🌐 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

    with open("chatbot_server.py", "w", encoding="utf-8") as f:
        f.write(flask_code)

    print("✅ Flask backend created: chatbot_server.py")

if __name__ == "__main__":
    # Choose interface type
    interface_type = input("Choose interface (1: Streamlit, 2: Console, 3: Web): ").strip()

    if interface_type == "1":
        print("🚀 Run with: streamlit run this_script.py")
        main()
    elif interface_type == "2":
        run_console_chatbot()
    elif interface_type == "3":
        create_simple_web_interface()
        create_flask_backend()
        print("🚀 Run with: python chatbot_server.py")
    else:
        print("📋 Available options:")
        print("1. Streamlit interface (recommended)")
        print("2. Console chatbot")
        print("3. Web interface with Flask")
        print("\n🚀 Default: Running Streamlit interface...")
        main()