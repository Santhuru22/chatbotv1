import streamlit as st
import os
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vislona AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Santhuru22/chatbotv1/issues',
        'Report a bug': 'https://github.com/Santhuru22/chatbotv1/issues/new',
        'About': """
        # Vislona AI Chatbot
        
        A powerful RAG-based chatbot powered by Ollama API and FAISS vector search.
        
        **Features:**
        - Intelligent context retrieval
        - Cloud-based LLM via Ollama API
        - Modern chat interface
        - Usage analytics
        - Export capabilities
        
        **GitHub:** https://github.com/Santhuru22/chatbotv1
        """
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main > div {
        padding: 1rem;
        max-width: 1200px;
        margin: auto;
    }
    
    .chat-container {
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        max-width: 80%;
    }
    
    .user-message {
        background: #e6f3ff;
        margin-left: auto;
        border: 1px solid #b8daff;
    }
    
    .assistant-message {
        background: #f8f9fa;
        margin-right: auto;
        border: 1px solid #dee2e6;
    }
    
    .github-header {
        background: linear-gradient(135deg, #24292e 0%, #0366d6 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
    }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-online {
        background: #d4edda;
        color: #155724;
    }
    
    .status-offline {
        background: #f8d7da;
        color: #721c24;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
        border-right: 1px solid #e1e4e8;
    }
    
    .source-reference {
        background: #f1f8ff;
        border-left: 4px solid #0366d6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    
    .stButton > button {
        background: #0366d6;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Environment configuration
class Config:
    """Configuration management for the chatbot"""
    
    def __init__(self):
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'https://api.ollama.com/v1')
        self.ollama_api_key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC37p+Suuw915wQt87anWmqk1GdrtEKz6bDnIKfqSbjx'
        self.default_embedding_model = os.getenv('DEFAULT_EMBEDDING_MODEL', 'nomic-embed-text')
        self.default_chat_model = os.getenv('DEFAULT_CHAT_MODEL', 'llama3.2:1b')
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH', 'faiss_index_ollama')
        self.max_context_chunks = int(os.getenv('MAX_CONTEXT_CHUNKS', '3'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.github_repo = os.getenv('GITHUB_REPO', 'Santhuru22/chatbotv1')
        self.version = os.getenv('APP_VERSION', '1.0.0')

config = Config()

# Import LangChain components with error handling
@st.cache_resource
def import_dependencies():
    """Import and cache dependencies"""
    try:
        from langchain.docstore.document import Document
        from langchain_community.vectorstores import FAISS
        from langchain.embeddings.base import Embeddings
        from langchain.llms.base import LLM
        return True, Document, FAISS, Embeddings, LLM, None
    except ImportError as e:
        error_msg = f"Missing dependencies: {str(e)}"
        logger.error(error_msg)
        return False, None, None, None, None, error_msg

# Check dependencies
deps_available, Document, FAISS, Embeddings, LLM, deps_error = import_dependencies()

# Custom Ollama API client
class OllamaAPIClient:
    """Client for interacting with Ollama cloud API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    def generate_embedding(self, text: str, model: str) -> List[float]:
        """Generate embeddings using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": model, "prompt": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return []
    
    def generate_response(self, prompt: str, model: str, temperature: float) -> str:
        """Generate response using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"Error generating response: {str(e)}"

# Custom Embeddings and LLM classes for Ollama API
class OllamaAPIEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str, api_key: str):
        super().__init__()
        self.model = model
        self.client = OllamaAPIClient(base_url, api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.generate_embedding(text, self.model) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self.client.generate_embedding(text, self.model)

class OllamaAPILLM(LLM):
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.client = OllamaAPIClient(base_url, api_key)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.client.generate_response(prompt, self.model, self.temperature)
    
    @property
    def _llm_type(self) -> str:
        return "ollama_api"

class GitHubVislonaRAG:
    """GitHub-optimized Vislona RAG Chatbot"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.is_initialized = False
    
    def initialize_components(self, embedding_model: str, chat_model: str, temperature: float):
        """Initialize LLM and embedding components"""
        try:
            logger.info(f"Initializing with models: {embedding_model}, {chat_model}")
            
            self.embeddings = OllamaAPIEmbeddings(
                model=embedding_model,
                base_url=self.config.ollama_base_url,
                api_key=self.config.ollama_api_key
            )
            
            self.llm = OllamaAPILLM(
                model=chat_model,
                base_url=self.config.ollama_base_url,
                api_key=self.config.ollama_api_key,
                temperature=temperature
            )
            
            self.is_initialized = True
            return True, "Components initialized successfully"
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False, f"Failed to initialize: {str(e)}"
    
    def load_or_create_vector_store(self):
        """Load existing vector store or create demo version"""
        vector_path = Path(self.config.vector_store_path)
        
        if vector_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True, f"Loaded existing vector store from {vector_path}"
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                return False, f"Failed to load vector store: {str(e)}"
        else:
            return self.create_demo_vector_store()
    
    def create_demo_vector_store(self):
        """Create a demo vector store with comprehensive Vislona content"""
        demo_documents = [
            {
                "content": """Vislona AI Platform - Complete Overview
                
                Vislona is an enterprise-grade AI platform designed for modern businesses to build, train, and deploy machine learning models at scale. The platform combines cutting-edge AutoML capabilities with traditional custom model development workflows.
                
                Core Features:
                - Automated model training with hyperparameter optimization
                - Multi-framework support (TensorFlow, PyTorch, Scikit-learn, XGBoost)
                - Real-time and batch deployment options
                - Comprehensive model monitoring and observability
                - Team collaboration tools with role-based access control
                - Enterprise security with SOC 2 compliance""",
                "metadata": {"source": "platform_overview", "category": "general", "priority": "high"}
            },
            {
                "content": """Vislona Model Training Capabilities
                
                Advanced Training Features:
                - AutoML: Automatically selects optimal algorithms and hyperparameters
                - Custom Training: Support for custom training scripts and containers
                - Distributed Training: Scale training across multiple GPUs and nodes
                - Transfer Learning: Leverage pre-trained models for faster training
                - Experiment Tracking: Version control for models, data, and experiments
                - Data Pipeline: Automated data preprocessing and feature engineering
                
                Supported Frameworks:
                - TensorFlow 2.x with Keras integration
                - PyTorch with Lightning support  
                - Scikit-learn for classical ML
                - XGBoost for gradient boosting
                - Custom Docker containers for any framework""",
                "metadata": {"source": "training_features", "category": "technical", "priority": "high"}
            },
            {
                "content": """Vislona Deployment and MLOps
                
                Deployment Options:
                - REST API endpoints with automatic scaling
                - Batch prediction jobs for large datasets
                - Real-time streaming for live predictions
                - Edge deployment for IoT and mobile devices
                - Serverless functions for event-driven predictions
                
                MLOps Features:
                - CI/CD pipelines for model deployment
                - A/B testing and canary deployments
                - Model versioning and rollback capabilities
                - Performance monitoring and alerting
                - Data drift detection and model retraining
                - Cost optimization with auto-scaling""",
                "metadata": {"source": "deployment_mlops", "category": "technical", "priority": "medium"}
            },
            {
                "content": """Vislona Team Collaboration
                
                Collaboration Tools:
                - Shared workspaces for team projects
                - Model registry for sharing and discovery
                - Jupyter notebook integration with real-time collaboration
                - Version control for datasets, models, and experiments
                - Code review workflows for ML projects
                - Integration with Git, GitHub, and GitLab
                
                Access Control:
                - Role-based permissions (Admin, Data Scientist, Viewer)
                - Project-level access control
                - API key management for secure access
                - Audit logs for compliance and security
                - Single Sign-On (SSO) integration""",
                "metadata": {"source": "collaboration", "category": "features", "priority": "medium"}
            },
            {
                "content": """Vislona Pricing and Plans
                
                Free Tier (Hobby):
                - Up to 100 API calls per month
                - Basic model training (2 hours compute time)
                - Community support
                - Public model sharing
                
                Professional ($49/month):
                - Unlimited API calls
                - Advanced AutoML features
                - Priority support (email)
                - Private models and datasets
                - Advanced monitoring and alerts
                
                Team ($199/month):
                - Everything in Professional
                - Multi-user workspaces
                - Advanced collaboration features
                - Dedicated compute resources
                - Phone and chat support
                
                Enterprise (Custom):
                - Custom pricing based on usage
                - Dedicated infrastructure and support
                - On-premise deployment options
                - SLA guarantees and custom contracts
                - Advanced security and compliance features""",
                "metadata": {"source": "pricing", "category": "business", "priority": "medium"}
            },
            {
                "content": """Vislona Security and Compliance
                
                Security Measures:
                - End-to-end encryption (AES-256) for data in transit and at rest
                - VPC deployment with private networking
                - Regular security audits and penetration testing
                - Vulnerability scanning and patch management
                - Backup and disaster recovery procedures
                
                Compliance Certifications:
                - SOC 2 Type II compliance
                - GDPR compliance for data privacy
                - HIPAA compliance for healthcare data
                - ISO 27001 information security management
                - Regular compliance audits and reporting
                
                Access Security:
                - Multi-factor authentication (MFA)
                - Single Sign-On (SSO) with SAML and OAuth
                - API rate limiting and DDoS protection
                - Comprehensive audit logging""",
                "metadata": {"source": "security_compliance", "category": "security", "priority": "high"}
            }
        ]
        
        try:
            documents = []
            for i, doc_data in enumerate(demo_documents):
                doc = Document(
                    page_content=doc_data["content"],
                    metadata={
                        **doc_data["metadata"],
                        "chunk_id": f"demo_{i}",
                        "created_at": datetime.now().isoformat(),
                        "version": config.version
                    }
                )
                documents.append(doc)
            
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            vector_path = Path(self.config.vector_store_path)
            vector_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(vector_path))
            
            logger.info(f"Created demo vector store with {len(documents)} documents")
            return True, f"Created demo knowledge base with {len(documents)} comprehensive documents"
        except Exception as e:
            logger.error(f"Error creating demo vector store: {e}")
            return False, f"Failed to create demo vector store: {str(e)}"
    
    def get_relevant_context(self, query: str, max_chunks: int):
        """Retrieve relevant context from vector store"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=max_chunks)
            context_chunks = []
            for doc, score in results:
                similarity = max(0, 1 - score)
                context_chunks.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "similarity": similarity,
                    "metadata": doc.metadata
                })
            return context_chunks
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, query: str, context_chunks: List[Dict], temperature: float):
        """Generate response using LLM with context"""
        if not self.llm or not self.is_initialized:
            return "Assistant is not properly initialized. Please check the configuration."
        
        if context_chunks:
            context_text = "\n\n".join([
                f"Reference {i+1} (Relevance: {chunk['similarity']:.1%}):\n{chunk['content']}"
                for i, chunk in enumerate(context_chunks)
            ])
            prompt = f"""You are Vislona AI Assistant, an expert AI consultant for the Vislona platform. Use the provided context to give comprehensive, accurate answers.

CONTEXT INFORMATION:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide detailed, professional responses based on the context
- Include specific features, capabilities, and benefits when relevant
- If the context doesn't fully cover the question, acknowledge this and provide general AI/ML guidance
- Use clear structure with headings or bullet points for complex topics
- Maintain a helpful, knowledgeable tone throughout

RESPONSE:"""
        else:
            prompt = f"""You are Vislona AI Assistant, a knowledgeable AI consultant. While I don't have specific context for this question, I'll provide helpful general guidance.

USER QUESTION: {query}

INSTRUCTIONS:
- Provide helpful information about AI, machine learning, or general assistance
- Be honest about limitations when specific Vislona information isn't available
- Offer practical advice and best practices
- Maintain a professional, supportive tone

RESPONSE:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error generating a response: {str(e)}. Please ensure the Ollama API is accessible."
    
    def chat(self, query: str, max_chunks: int, temperature: float):
        """Main chat function"""
        start_time = time.time()
        context_chunks = self.get_relevant_context(query, max_chunks)
        response = self.generate_response(query, context_chunks, temperature)
        response_time = time.time() - start_time
        
        return {
            "response": response,
            "context_chunks": context_chunks,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "chat_model": getattr(self.llm, 'model', 'unknown'),
                "embedding_model": getattr(self.embeddings, 'model', 'unknown')
            }
        }

@st.cache_data(ttl=30)
def check_ollama_status(base_url: str, api_key: str):
    """Check Ollama API status and available models"""
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, [model['name'] for model in models]
        return False, []
    except Exception as e:
        logger.error(f"Ollama API connection error: {e}")
        return False, []

def display_github_header():
    """Display GitHub-style header"""
    st.markdown(f"""
    <div class="github-header">
        <h1>🤖 Vislona AI Chatbot</h1>
        <p>Cloud-powered AI with RAG Technology | Version {config.version}</p>
        <p><strong>Repository:</strong> <a href="https://github.com/{config.github_repo}" target="_blank" style="color: #ffd700;">github.com/{config.github_repo}</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin: 1rem 0;'>
        <img src="https://img.shields.io/github/stars/Santhuru22/chatbotv1?style=social" alt="GitHub stars">
        <img src="https://img.shields.io/github/issues/Santhuru22/chatbotv1" alt="Issues">
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display configuration sidebar"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ollama_online, available_models = check_ollama_status(config.ollama_base_url, config.ollama_api_key)
        
        if ollama_online:
            st.markdown(f'<div class="status-badge status-online">✓ Ollama API Online ({len(available_models)} models)</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-offline">✗ Ollama API Offline</div>', 
                       unsafe_allow_html=True)
            st.error("Please verify the Ollama API key and endpoint.")
            return None
        
        st.divider()
        
        embedding_models = [m for m in available_models if 'embed' in m.lower()]
        chat_models = [m for m in available_models if not 'embed' in m.lower()]
        
        embedding_model = st.selectbox(
            "🔍 Embedding Model",
            options=embedding_models or [config.default_embedding_model],
            index=0,
            help="Model for creating vector embeddings"
        )
        
        chat_model = st.selectbox(
            "💬 Chat Model", 
            options=chat_models or [config.default_chat_model],
            index=0,
            help="Model for generating responses"
        )
        
        with st.expander("⚙️ Advanced Settings"):
            max_chunks = st.slider("Max Context Chunks", 1, 5, config.max_context_chunks)
            temperature = st.slider("Temperature", 0.0, 1.0, config.temperature, 0.1)
        
        return {
            "embedding_model": embedding_model,
            "chat_model": chat_model,
            "max_chunks": max_chunks,
            "temperature": temperature
        }

def display_features():
    """Display key features"""
    st.subheader("✨ Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Smart RAG</h4>
            <p>Context-aware responses using vector search and retrieval augmented generation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>☁️ Cloud-Powered</h4>
            <p>Seamless integration with Ollama's cloud API for scalable AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Analytics</h4>
            <p>Real-time usage statistics and source attribution</p>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface(chatbot: GitHubVislonaRAG, settings: dict):
    """Display modern chat interface"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"""👋 **Welcome to Vislona AI Assistant!**

I'm your cloud-powered AI assistant for the Vislona platform. I can help with:

- **Platform Features**: Explore training, deployment, and collaboration tools
- **Technical Guidance**: Get AI/ML best practices and insights
- **Pricing & Plans**: Understand subscription options
- **Security & Compliance**: Learn about data protection

*Powered by Ollama API and RAG technology for accurate, contextual responses.*

What would you like to know about Vislona today?""",
            "timestamp": datetime.now().isoformat()
        }]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.markdown(f'<div class="{message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)
            
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                cols = st.columns(4)
                cols[0].caption(f"⏱️ {metadata.get('response_time', 0):.2f}s")
                cols[1].caption(f"📄 {len(metadata.get('context_chunks', []))} sources")
                cols[2].caption(f"🔧 {metadata.get('model_info', {}).get('chat_model', 'unknown')}")
                cols[3].caption(f"🕒 {datetime.fromisoformat(message['timestamp']).strftime('%H:%M:%S')}")
                
                if metadata.get('context_chunks'):
                    with st.expander("📚 Sources"):
                        for i, chunk in enumerate(metadata['context_chunks'], 1):
                            st.markdown(f"""
                            <div class="source-reference">
                                <strong>Source {i}</strong> (Relevance: {chunk['similarity']:.1%})<br>
                                <small>Category: {chunk['metadata'].get('category', 'Unknown')} | 
                                Priority: {chunk['metadata'].get('priority', 'Normal')}</small>
                                <hr style="margin: 0.5rem 0;">
                                {chunk['content'][:400]}{'...' if len(chunk['content']) > 400 else ''}
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if user_input := st.chat_input("Ask about Vislona AI..."):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Generating response..."):
                result = chatbot.chat(
                    user_input, 
                    settings["max_chunks"], 
                    settings["temperature"]
                )
                
                st.markdown(f'<div class="assistant-message">{result["response"]}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "timestamp": result["timestamp"],
                    "metadata": result
                })
                
                st.rerun()

def display_usage_stats():
    """Display usage statistics"""
    if len(st.session_state.get("messages", [])) > 1:
        assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant" and "metadata" in m]
        if assistant_messages:
            total_queries = len(assistant_messages)
            avg_response_time = sum(m["metadata"]["response_time"] for m in assistant_messages) / total_queries
            total_sources = sum(len(m["metadata"]["context_chunks"]) for m in assistant_messages)
            
            cols = st.columns(3)
            cols[0].metric("Total Queries", total_queries)
            cols[1].metric("Avg Response Time", f"{avg_response_time:.2f}s")
            cols[2].metric("Sources Used", total_sources)

def main():
    """Main application function"""
    if not deps_available:
        st.error("❌ Required dependencies not available!")
        st.code(f"Error: {deps_error}")
        st.info("Install with: `pip install -r requirements.txt`")
        with st.expander("📋 Required Dependencies"):
            st.code("""
streamlit>=1.28.0
langchain>=0.3.0
langchain-community>=0.3.0
faiss-cpu>=1.7.4
pandas>=1.5.0
requests>=2.31.0
""")
        st.stop()
    
    display_github_header()
    display_features()
    
    settings = display_sidebar()
    if not settings:
        st.warning("⚠️ Please configure Ollama API first")
        st.stop()
    
    if "chatbot" not in st.session_state:
        with st.spinner("🚀 Initializing Vislona AI Assistant..."):
            chatbot = GitHubVislonaRAG(config)
            success, message = chatbot.initialize_components(
                settings["embedding_model"],
                settings["chat_model"], 
                settings["temperature"]
            )
            
            if not success:
                st.error(f"❌ Initialization failed: {message}")
                st.stop()
            
            vs_success, vs_message = chatbot.load_or_create_vector_store()
            if vs_success:
                st.success(f"✅ {vs_message}")
            else:
                st.error(f"❌ Vector store error: {vs_message}")
                st.stop()
            
            st.session_state.chatbot = chatbot
            st.success("🎉 Vislona AI Assistant is ready!")
    
    st.subheader("💬 Chat with Vislona AI")
    display_chat_interface(st.session_state.chatbot, settings)
    
    with st.sidebar:
        st.divider()
        st.subheader("📊 Usage Statistics")
        display_usage_stats()
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset"):
                for key in ["chatbot", "messages"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        if len(st.session_state.get("messages", [])) > 2:
            st.divider()
            assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant" and "metadata" in m]
            if assistant_messages:
                export_data = []
                for i, msg in enumerate(assistant_messages):
                    user_msg = st.session_state.messages[st.session_state.messages.index(msg) - 1]
                    export_data.append({
                        "timestamp": msg["timestamp"],
                        "user_query": user_msg["content"],
                        "response": msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"],
                        "response_time": msg["metadata"]["response_time"],
                        "sources_used": len(msg["metadata"]["context_chunks"]),
                        "chat_model": msg["metadata"]["model_info"]["chat_model"],
                        "embedding_model": msg["metadata"]["model_info"]["embedding_model"]
                    })
                
                df = pd.DataFrame(export_data)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    "📥 Export Chat History",
                    data=csv_data,
                    file_name=f"vislona_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download chat history as CSV file"
                )
        
        st.divider()
        st.subheader("🔗 GitHub Links")
        st.markdown(f"""
        - [📖 Repository](https://github.com/{config.github_repo})
        - [🐛 Report Issues](https://github.com/{config.github_repo}/issues)
        - [📋 Documentation](https://github.com/{config.github_repo}#readme)
        - [⭐ Star the Repo](https://github.com/{config.github_repo})
        """)
        
        with st.expander("🔧 System Info"):
            st.json({
                "app_version": config.version,
                "ollama_url": config.ollama_base_url,
                "vector_store": config.vector_store_path,
                "models_configured": bool(settings),
                "chat_initialized": "chatbot" in st.session_state
            })

if __name__ == "__main__":
    main()
