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
        
        A powerful RAG-based chatbot powered by Ollama and FAISS vector search.
        
        **Features:**
        - Intelligent context retrieval
        - Multiple LLM support via Ollama
        - Real-time chat interface
        - Usage analytics
        - Export capabilities
        
        **GitHub:** https://github.com/Santhuru22/chatbotv1
        """
    }
)

# Custom CSS with GitHub-friendly styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    .github-header {
        background: linear-gradient(135deg, #24292e 0%, #0366d6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #e1e4e8;
    }
    
    .feature-card {
        background: #f6f8fa;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-online {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-offline {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .github-button {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.75rem;
        background: #0366d6;
        color: white;
        border-radius: 6px;
        text-decoration: none;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    
    .source-reference {
        background: #f1f8ff;
        border: 1px solid #c9e6ff;
        border-left: 4px solid #0366d6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# Environment configuration
class Config:
    """Configuration management for the chatbot"""
    
    def __init__(self):
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.default_embedding_model = os.getenv('DEFAULT_EMBEDDING_MODEL', 'nomic-embed-text')
        self.default_chat_model = os.getenv('DEFAULT_CHAT_MODEL', 'llama3.2:1b')
        self.vector_store_path = os.getenv('VECTOR_STORE_PATH', 'faiss_index_ollama')
        self.max_context_chunks = int(os.getenv('MAX_CONTEXT_CHUNKS', '3'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY', '639a1643fecc4ecf8198f79c4f04b22b.yZJbvn99TEMQwzJ-hPXfRtu6')  # Added API key
        
        # GitHub specific settings
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
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        return True, Document, FAISS, OllamaEmbeddings, OllamaLLM, None
    except ImportError as e:
        error_msg = f"Missing dependencies: {str(e)}"
        logger.error(error_msg)
        return False, None, None, None, None, error_msg

# Check dependencies
deps_available, Document, FAISS, OllamaEmbeddings, OllamaLLM, deps_error = import_dependencies()

class GitHubVislonaRAG:
    """GitHub-optimized Vislona RAG Chatbot"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.is_initialized = False
        
    def initialize_components(self, embedding_model: str, chat_model: str, temperature: float):
        """Initialize LLM and embedding components with API key"""
        try:
            logger.info(f"Initializing with models: {embedding_model}, {chat_model}")
            
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=self.config.ollama_base_url,
                headers={"Authorization": f"Bearer {self.config.ollama_api_key}" if self.config.ollama_api_key else None}  # Added API key header
            )
            
            self.llm = OllamaLLM(
                model=chat_model,
                base_url=self.config.ollama_base_url,
                temperature=temperature,
                headers={"Authorization": f"Bearer {self.config.ollama_api_key}" if self.config.ollama_api_key else None}  # Added API key header
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
            
            # Create vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
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
                similarity = max(0, 1 - score)  # Convert distance to similarity
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
        
        # Create context-aware prompt
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
            return f"I encountered an error generating a response: {str(e)}. Please ensure Ollama is running and the model is available."
    
    def chat(self, query: str, max_chunks: int, temperature: float):
        """Main chat function"""
        start_time = time.time()
        
        # Get relevant context
        context_chunks = self.get_relevant_context(query, max_chunks)
        
        # Generate response
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
def check_ollama_status(base_url: str = None):
    """Check Ollama server status and available models"""
    url = base_url or config.ollama_base_url
    headers = {"Authorization": f"Bearer {config.ollama_api_key}"} if config.ollama_api_key else {}
    try:
        response = requests.get(f"{url}/api/tags", headers=headers, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, [model['name'] for model in models]
        return False, []
    except Exception as e:
        logger.error(f"Ollama connection error: {e}")
        return False, []

def display_github_header():
    """Display GitHub-style header"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="github-header">
            <h1>🤖 Vislona AI Chatbot</h1>
            <p>Powered by Ollama & RAG Technology | Version {config.version}</p>
            <p><strong>Repository:</strong> <a href="https://github.com/{config.github_repo}" target="_blank" style="color: #ffd700;">github.com/{config.github_repo}</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # GitHub stats/badges could go here
        st.markdown("""
        [![GitHub stars](https://img.shields.io/github/stars/Santhuru22/chatbotv1?style=social)](https://github.com/Santhuru22/chatbotv1)
        
        [![Issues](https://img.shields.io/github/issues/Santhuru22/chatbotv1)](https://github.com/Santhuru22/chatbotv1/issues)
        """)

def display_sidebar():
    """Display configuration sidebar"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama status
        ollama_online, available_models = check_ollama_status()
        
        if ollama_online:
            st.markdown(f'<div class="status-badge status-online">✓ Ollama Online ({len(available_models)} models)</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-offline">✗ Ollama Offline</div>', 
                       unsafe_allow_html=True)
            
            with st.expander("🚀 Setup Instructions", expanded=True):
                st.markdown("""
                **1. Ensure Remote Ollama is Running:**
                Contact your server administrator to confirm the Ollama instance is active.
                
                **2. Configure URL:**
                Use the sidebar to set the correct Ollama Base URL.
                """)
            return None
        
        st.divider()
        
        # Model selection
        embedding_models = [m for m in available_models if 'embed' in m.lower()]
        chat_models = [m for m in available_models if not 'embed' in m.lower()]
        
        embedding_model = st.selectbox(
            "🔍 Embedding Model",
            options=embedding_models or available_models[:2],
            index=0,
            help="Model for creating vector embeddings"
        )
        
        chat_model = st.selectbox(
            "💬 Chat Model", 
            options=chat_models or available_models,
            index=0,
            help="Model for generating responses"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            max_chunks = st.slider("Max Context Chunks", 1, 5, 3)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            
            st.text_input(
                "Ollama Base URL",
                value=config.ollama_base_url,
                help="Ollama server endpoint",
                key="ollama_url"
            )
            st.text_input(
                "Ollama API Key",
                value=config.ollama_api_key,
                help="API key for authenticated access (if required)",
                type="password",
                key="ollama_api_key"
            )
        
        return {
            "embedding_model": embedding_model,
            "chat_model": chat_model,
            "max_chunks": max_chunks,
            "temperature": temperature,
            "ollama_api_key": st.session_state.get("ollama_api_key", config.ollama_api_key)
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
            <h4>📊 Analytics</h4>
            <p>Real-time usage statistics, response times, and source attribution</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>⚙️ Configurable</h4>
            <p>Multiple LLM support via Ollama with customizable parameters</p>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface(chatbot: GitHubVislonaRAG, settings: dict):
    """Display main chat interface"""
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": f"""👋 **Welcome to Vislona AI Assistant!**

I'm your AI-powered assistant for the Vislona platform. I can help you with:

- **Platform Features**: Learn about training, deployment, and collaboration tools
- **Technical Guidance**: Get help with AI/ML concepts and best practices  
- **Pricing & Plans**: Understand different subscription options
- **Security & Compliance**: Information about data protection and certifications

*This chatbot uses RAG technology to provide accurate, contextual responses. Source attribution is shown for transparency.*

What would you like to know about Vislona today?""",
            "timestamp": datetime.now().isoformat()
        }]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.caption(f"⏱️ {metadata.get('response_time', 0):.2f}s")
                with col2:
                    st.caption(f"📄 {len(metadata.get('context_chunks', []))} sources")
                with col3:
                    st.caption(f"🔧 {metadata.get('model_info', {}).get('chat_model', 'unknown')}")
                with col4:
                    timestamp = datetime.fromisoformat(message["timestamp"])
                    st.caption(f"🕒 {timestamp.strftime('%H:%M:%S')}")
                
                # Show sources
                if metadata.get('context_chunks'):
                    with st.expander("📚 View Sources", expanded=False):
                        for i, chunk in enumerate(metadata['context_chunks'], 1):
                            st.markdown(f"""
                            <div class="source-reference">
                                <strong>Source {i}</strong> (Relevance: {chunk['similarity']:.1%})<br>
                                <small>Category: {chunk['metadata'].get('category', 'Unknown')} | 
                                Priority: {chunk['metadata'].get('priority', '
