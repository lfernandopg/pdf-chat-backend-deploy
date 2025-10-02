"""
Repository layer for vector store operations
Handles low-level interactions with FAISS and file system
"""
from pathlib import Path
from typing import Optional, List
import json
import shutil
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from app.core.config import settings
from app.models.schemas import ChatMetadata


class VectorStoreRepository:
    """Repository for managing vector stores and metadata"""
    
    def __init__(self):
        self.storage_path = Path(settings.STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_chat_path(self, chat_id: str) -> Path:
        """Get the directory path for a specific chat"""
        return self.storage_path / chat_id
    
    def _get_vector_store_path(self, chat_id: str) -> Path:
        """Get the path for vector store files"""
        return self._get_chat_path(chat_id) / "vector_store"
    
    def _get_metadata_path(self, chat_id: str) -> Path:
        """Get the path for metadata file"""
        return self._get_chat_path(chat_id) / "metadata.json"
    
    def _get_documents_path(self, chat_id: str) -> Path:
        """Get the path for stored documents"""
        return self._get_chat_path(chat_id) / "documents"
    
    def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat exists"""
        chat_path = self._get_chat_path(chat_id)
        vector_store_path = self._get_vector_store_path(chat_id)
        metadata_path = self._get_metadata_path(chat_id)
        
        return (
            chat_path.exists() and
            vector_store_path.exists() and
            metadata_path.exists()
        )
    
    def create_vector_store(
        self,
        chat_id: str,
        documents: List[Document],
        embedding_model_name: str
    ) -> FAISS:
        """Create a new FAISS vector store"""
        # Create chat directory
        chat_path = self._get_chat_path(chat_id)
        chat_path.mkdir(parents=True, exist_ok=True)
        
        # Create documents directory
        docs_path = self._get_documents_path(chat_id)
        docs_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save vector store
        vector_store_path = self._get_vector_store_path(chat_id)
        vector_store.save_local(str(vector_store_path))
        
        return vector_store
    
    def load_vector_store(
        self,
        chat_id: str,
        embedding_model_name: str
    ) -> Optional[FAISS]:
        """Load an existing FAISS vector store"""
        if not self.chat_exists(chat_id):
            return None
        
        vector_store_path = self._get_vector_store_path(chat_id)
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        vector_store = FAISS.load_local(
            str(vector_store_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vector_store
    
    def add_documents_to_vector_store(
        self,
        chat_id: str,
        new_documents: List[Document],
        embedding_model_name: str
    ) -> FAISS:
        """Add new documents to an existing vector store"""
        # Load existing vector store
        vector_store = self.load_vector_store(chat_id, embedding_model_name)
        
        if vector_store is None:
            raise ValueError(f"Chat {chat_id} does not exist")
        
        # Add new documents
        vector_store.add_documents(new_documents)
        
        # Save updated vector store
        vector_store_path = self._get_vector_store_path(chat_id)
        vector_store.save_local(str(vector_store_path))
        
        return vector_store
    
    def save_metadata(self, chat_id: str, metadata: ChatMetadata) -> None:
        """Save chat metadata to JSON file"""
        metadata_path = self._get_metadata_path(chat_id)
        
        with open(metadata_path, 'w') as f:
            json.dump(
                metadata.model_dump(mode='json'),
                f,
                indent=2,
                default=str
            )
    
    def load_metadata(self, chat_id: str) -> Optional[ChatMetadata]:
        """Load chat metadata from JSON file"""
        metadata_path = self._get_metadata_path(chat_id)
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return ChatMetadata(**data)
    
    def update_metadata(
        self,
        chat_id: str,
        documents_count: Optional[int] = None,
        chunks_count: Optional[int] = None
    ) -> None:
        """Update specific fields in metadata"""
        metadata = self.load_metadata(chat_id)
        
        if metadata is None:
            raise ValueError(f"Metadata for chat {chat_id} not found")
        
        if documents_count is not None:
            metadata.documents_count = documents_count
        
        if chunks_count is not None:
            metadata.chunks_count = chunks_count
        
        metadata.last_updated = datetime.utcnow()
        
        self.save_metadata(chat_id, metadata)
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete all files and directories for a chat"""
        chat_path = self._get_chat_path(chat_id)
        
        if not chat_path.exists():
            return False
        
        try:
            shutil.rmtree(chat_path)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete chat {chat_id}: {str(e)}")
    
    def save_uploaded_file(
        self,
        chat_id: str,
        filename: str,
        content: bytes
    ) -> Path:
        """Save an uploaded file to the documents directory"""
        docs_path = self._get_documents_path(chat_id)
        docs_path.mkdir(parents=True, exist_ok=True)
        
        file_path = docs_path / filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return file_path
    
    def get_vector_store_size(self, chat_id: str) -> int:
        """Get the number of vectors in the store"""
        metadata = self.load_metadata(chat_id)
        return metadata.chunks_count if metadata else 0