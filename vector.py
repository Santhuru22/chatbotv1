```python
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.docstore.document import Document
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LangChainSplitter
    LANGCHAIN_AVAILABLE = True
    logger.info("✅ LangChain libraries loaded successfully")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.error(f"❌ LangChain not available: {e}")
    logger.info("Install with: pip install langchain langchain-community faiss-cpu")

class VislonaVectorStoreProcessor:
    """
    Process saved Vislona chunks and create FAISS vector store for semantic search using Ollama
    """
    
    def __init__(self, 
                 chunks_directory: str = os.getenv("CHUNKS_DIRECTORY", "./dataset/chunks"),
                 ollama_model: str = os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
                 ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")):
        """
        Initialize the vector store processor
        
        Args:
            chunks_directory: Directory containing saved chunk files
            ollama_model: Ollama embedding model to use (default: nomic-embed-text)
            ollama_base_url: Ollama server URL (default: http://localhost:11434 or env var)
        """
        self.chunks_directory = Path(chunks_directory)
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", None)  # Support for API key if needed
        
        # Create chunks directory if it doesn't exist
        self.chunks_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🦙 Using Ollama model: {ollama_model}")
        logger.info(f"🌐 Ollama server: {ollama_base_url}")
    
    def load_chunks_from_files(self) -> List[Dict[str, Any]]:
        """
        Load all chunk files from the chunks directory
        
        Returns:
            List of dictionaries containing chunk data
        """
        if not self.chunks_directory.exists():
            logger.error(f"❌ Chunks directory not found: {self.chunks_directory}")
            return []
        
        chunk_files = list(self.chunks_directory.glob("chunk_*.txt"))
        chunk_files.sort()  # Sort by filename for consistent ordering
        
        chunks_data = []
        
        logger.info(f"📁 Loading {len(chunk_files)} chunk files...")
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse chunk metadata from file header
                metadata = self._parse_chunk_metadata(content)
                
                # Extract actual content (remove header)
                actual_content = self._extract_chunk_content(content)
                
                if actual_content.strip():  # Only add non-empty chunks
                    chunk_data = {
                        "content": actual_content,
                        "file_path": str(chunk_file),
                        "file_name": chunk_file.name,
                        **metadata
                    }
                    chunks_data.append(chunk_data)
                    
            except Exception as e:
                logger.error(f"❌ Error loading {chunk_file}: {e}")
                continue
        
        logger.info(f"✅ Successfully loaded {len(chunks_data)} chunks")
        return chunks_data
    
    def _parse_chunk_metadata(self, content: str) -> Dict[str, Any]:
        """Parse metadata from chunk file header"""
        metadata = {}
        
        # Extract chunk ID
        chunk_id_match = re.search(r'# Chunk (\d+)', content)
        if chunk_id_match:
            metadata['chunk_id'] = int(chunk_id_match.group(1))
        
        # Extract source
        source_match = re.search(r'# Source: (.+)', content)
        if source_match:
            metadata['source'] = source_match.group(1).strip()
        
        # Extract position
        position_match = re.search(r'# Position: (\d+)-(\d+)', content)
        if position_match:
            metadata['start_index'] = int(position_match.group(1))
            metadata['end_index'] = int(position_match.group(2))
        
        # Extract length
        length_match = re.search(r'# Length: (\d+) characters', content)
        if length_match:
            metadata['chunk_size'] = int(length_match.group(1))
        
        # Add default metadata
        metadata.update({
            "source_type": "vislona_chatbot_dataset",
            "document_type": "chatbot_training_data",
            "company": "Vislona AI Platform",
            "processing_date": "2025-09-04"
        })
        
        return metadata
    
    def _extract_chunk_content(self, content: str) -> str:
        """Extract actual content from chunk file (remove header)"""
        lines = content.split('\n')
        content_start = 0
        
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip() == '':
                content_start = i + 1
                break
            elif not line.startswith('#') and line.strip():
                content_start = i
                break
        
        return '\n'.join(lines[content_start:]).strip()
    
    def create_langchain_documents(self, chunks_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert chunk data to LangChain Document objects
        
        Args:
            chunks_data: List of chunk dictionaries
            
        Returns:
            List of LangChain Document objects
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        documents = []
        
        for chunk_data in chunks_data:
            metadata = {k: v for k, v in chunk_data.items() if k != 'content'}
            doc = Document(
                page_content=chunk_data['content'],
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"✅ Created {len(documents)} LangChain Document objects")
        return documents
    
    def test_ollama_connection(self) -> Tuple[bool, List[str]]:
        """
        Test connection to Ollama server and check available models
        
        Returns:
            Tuple of (connection_success, available_models)
        """
        try:
            headers = {}
            if self.ollama_api_key:
                headers["Authorization"] = f"Bearer {self.ollama_api_key}"
            
            response = requests.get(f"{self.ollama_base_url}/api/tags", headers=headers, timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                logger.info(f"🦙 Available Ollama models: {', '.join(model_names)}")
                
                if self.ollama_model in model_names:
                    logger.info(f"✅ Embedding model '{self.ollama_model}' is available")
                    return True, model_names
                else:
                    logger.warning(f"⚠️ Embedding model '{self.ollama_model}' not found")
                    embedding_models = [
                        "nomic-embed-text", "mxbai-embed-large", "all-minilm",
                        "snowflake-arctic-embed", "bge-large", "bge-base"
                    ]
                    available_embedding_models = [m for m in model_names if any(em in m for em in embedding_models)]
                    
                    if available_embedding_models:
                        logger.info(f"✅ Found alternative embedding models: {', '.join(available_embedding_models)}")
                        self.ollama_model = available_embedding_models[0]
                        logger.info(f"🔄 Switching to: {self.ollama_model}")
                        return True, model_names
                    else:
                        logger.info(f"💡 Pull the recommended model with: ollama pull {self.ollama_model}")
                        return False, model_names
            else:
                logger.error(f"❌ Failed to connect to Ollama server at {self.ollama_base_url}: {response.status_code} {response.text}")
                return False, []
                
        except Exception as e:
            logger.error(f"❌ Error testing Ollama connection: {e}")
            logger.info("💡 Make sure Ollama is running or check OLLAMA_BASE_URL and OLLAMA_API_KEY")
            return False, []
    
    def create_faiss_vectorstore(self, 
                                documents: List[Document], 
                                save_path: str = os.getenv("VECTOR_STORE_PATH", "faiss_index_ollama")) -> Optional[FAISS]:
        """
        Create FAISS vector store from documents using Ollama embeddings
        
        Args:
            documents: List of LangChain Document objects
            save_path: Path to save the vector store
            
        Returns:
            FAISS vector store or None if failed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        # Test Ollama connection
        connection_success, available_models = self.test_ollama_connection()
        if not connection_success:
            logger.error("❌ Cannot proceed without Ollama connection")
            return None
        
        try:
            logger.info(f"🔍 Creating FAISS vector store with Ollama embeddings...")
            
            # Create Ollama embeddings
            embeddings = OllamaEmbeddings(
                model=self.ollama_model,
                base_url=self.ollama_base_url
            )
            
            # Test embeddings
            logger.info("🧪 Testing embeddings...")
            test_embedding = embeddings.embed_query("test query")
            logger.info(f"✅ Embedding dimension: {len(test_embedding)}")
            
            # Create FAISS vector store
            logger.info("🏗️ Building vector store...")
            vector_store = FAISS.from_documents(documents, embeddings)
            
            # Save vector store
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(save_path)
            logger.info(f"💾 Vector store saved to: {save_path}")
            
            # Test loading
            logger.info("🧪 Testing vector store loading...")
            test_vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"✅ Vector store created and tested successfully!")
            logger.info(f"📊 Total documents in vector store: {len(documents)}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {e}")
            if "connection" in str(e).lower():
                logger.info("💡 Make sure Ollama is running or check OLLAMA_BASE_URL")
            elif "model" in str(e).lower():
                logger.info(f"💡 Make sure the model is available: ollama pull {self.ollama_model}")
            return None
    
    def query_vectorstore(self, 
                         vector_store: FAISS, 
                         query: str, 
                         k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents
        
        Args:
            vector_store: FAISS vector store
            query: Query string
            k: Number of results to return
            
        Returns:
            List of results with content, score, and metadata
        """
        try:
            results = vector_store.similarity_search_with_score(query, k=k)
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata,
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error querying vector store: {e}")
            return []
    
    def run_sample_queries(self, vector_store: FAISS) -> None:
        """Run sample queries on the vector store"""
        sample_queries = [
            "What is Vislona?",
            "How do I train AI models?",
            "What are the pricing plans?",
            "How does team collaboration work?",
            "What file formats are supported?",
            "How do I deploy my model?",
            "What security features are available?",
            "How do I get an internship?"
        ]
        
        logger.info("\n🔍 Running Sample Queries:")
        logger.info("=" * 60)
        
        for i, query in enumerate(sample_queries, 1):
            logger.info(f"\n❓ Query {i}: {query}")
            logger.info("-" * 40)
            
            results = self.query_vectorstore(vector_store, query, k=2)
            
            for j, result in enumerate(results, 1):
                content_preview = result['content'][:200].replace('\n', ' ')
                logger.info(f"📄 Result {j} (Score: {result['similarity_score']:.4f}, Chunk: {result['chunk_id']}):")
                logger.info(f"   {content_preview}{'...' if len(result['content']) > 200 else ''}")
            
            if not results:
                logger.info("   No results found")

def main():
    """
    Main function to process Vislona chunks and create vector store using Ollama
    """
    logger.info("🚀 Starting Vislona Vector Store Creation Process (Ollama)")
    logger.info("=" * 60)
    
    processor = VislonaVectorStoreProcessor()
    chunks_data = processor.load_chunks_from_files()
    
    if not chunks_data:
        logger.error("❌ No chunks found. Make sure chunks are saved in the chunks directory.")
        return
    
    documents = processor.create_langchain_documents(chunks_data)
    vector_store = processor.create_faiss_vectorstore(documents)
    
    if vector_store:
        processor.run_sample_queries(vector_store)
        logger.info("\n🎉 Vector Store Creation Complete!")
        logger.info(f"📊 {len(documents)} documents processed")
        logger.info(f"💾 Vector store saved as '{os.getenv('VECTOR_STORE_PATH', 'faiss_index_ollama')}'")
        logger.info("🔍 Ready for semantic search queries")
        
        logger.info("\n💡 Usage Example:")
        logger.info("=" * 30)
        logger.info("from langchain_community.embeddings import OllamaEmbeddings")
        logger.info("from langchain.vectorstores import FAISS")
        logger.info("")
        logger.info("# Load the vector store")
        logger.info(f"embeddings = OllamaEmbeddings(model='{processor.ollama_model}', base_url='{processor.ollama_base_url}')")
        logger.info(f"vector_store = FAISS.load_local('{os.getenv('VECTOR_STORE_PATH', 'faiss_index_ollama')}', embeddings, allow_dangerous_deserialization=True)")
        logger.info("")
        logger.info("# Query the vector store")
        logger.info("results = vector_store.similarity_search('What is Vislona?', k=3)")
        logger.info("for result in results:")
        logger.info("    print(result.page_content)")
    else:
        logger.error("❌ Vector store creation failed")

def query_vislona_vectorstore(query: str, 
                             k: int = 3, 
                             vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "faiss_index_ollama"),
                             ollama_model: str = os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
                             ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")):
    """
    Query the Vislona vector store
    
    Args:
        query: Question to ask
        k: Number of results
        vector_store_path: Path to saved vector store
        ollama_model: Ollama embedding model
        ollama_base_url: Ollama server URL
    """
    if not LANGCHAIN_AVAILABLE:
        logger.error("❌ LangChain not available")
        return
    
    try:
        embeddings = OllamaEmbeddings(model=ollama_model, base_url=ollama_base_url)
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        logger.info(f"🔍 Query: {query}")
        logger.info("=" * 50)
        
        results = vector_store.similarity_search_with_score(query, k=k)
        
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"\n📄 Result {i} (Score: {score:.4f}):")
            logger.info(f"Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}")
            logger.info(f"Content: {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")
            logger.info("-" * 30)
            
    except Exception as e:
        logger.error(f"❌ Error querying vector store: {e}")
        if "connection" in str(e).lower():
            logger.info("💡 Make sure Ollama is running or check OLLAMA_BASE_URL")
        elif "model" in str(e).lower():
            logger.info(f"💡 Make sure the model is available: ollama pull {ollama_model}")

def check_ollama_models(base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                       api_key: Optional[str] = os.getenv("OLLAMA_API_KEY", None)):
    """
    Check available Ollama models
    """
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info("🦙 Available Ollama models:")
            for model in models:
                logger.info(f"   • {model['name']} ({model.get('size', 'unknown size')})")
        else:
            logger.error(f"❌ Cannot connect to Ollama server: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"❌ Error checking Ollama models: {e}")

def setup_ollama_embeddings():
    """
    Set up recommended embedding models for Ollama
    """
    logger.info("🦙 Setting up Ollama for embeddings...")
    logger.info("\n📋 Recommended embedding models:")
    logger.info("1. nomic-embed-text - General purpose text embeddings (recommended)")
    logger.info("2. mxbai-embed-large - Large embedding model for better accuracy")
    logger.info("3. all-minilm - Lightweight option")
    logger.info("4. snowflake-arctic-embed - High-quality embeddings")
    logger.info("\n💡 To install the recommended model, run:")
    logger.info("   ollama pull nomic-embed-text")
    logger.info("\n🚀 To start Ollama server:")
    logger.info("   ollama serve")
    logger.info("\n⚡ Quick setup commands:")
    logger.info("   ollama pull nomic-embed-text")
    logger.info("   # Then run your Python script")

def use_existing_model_for_embeddings(model_name: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")):
    """
    Use an existing chat model for embeddings as a fallback
    """
    logger.info(f"\n🔄 Alternative: Using {model_name} for embeddings")
    logger.warning("⚠️ Chat models aren't optimized for embeddings, but this can work as a temporary solution")
    
    processor = VislonaVectorStoreProcessor(
        ollama_model=model_name,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    return processor

if __name__ == "__main__":
    logger.info("🔧 Checking Ollama setup...")
    check_ollama_models()
    logger.info("\n")
    main()
```
