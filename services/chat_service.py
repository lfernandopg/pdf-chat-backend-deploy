"""
Service layer for chat operations
Contains business logic for RAG operations
"""
import uuid
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.repositories.vector_store_repository import VectorStoreRepository
from app.models.schemas import (
    ChatCreateResponse,
    ChatStatusResponse,
    AddDocumentsResponse,
    DeleteChatResponse,
    AskResponse,
    ChatMetadata
)
from app.core.config import settings


class ChatService:
    """Service for managing RAG chat operations"""
    
    def __init__(self, repository: VectorStoreRepository):
        self.repository = repository
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
    
    def _generate_chat_id(self) -> str:
        """Generate a unique chat ID"""
        return f"chat_{uuid.uuid4().hex[:12]}"
    
    def _validate_pdf_file(self, file: UploadFile) -> None:
        """Validate PDF file"""
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # Note: file.size might not be available in all cases
        # This is a basic check
        if hasattr(file, 'size') and file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
            )
    
    async def _process_pdf_files(
        self,
        files: List[UploadFile]
    ) -> tuple[List, int]:
        """Process PDF files and extract text chunks"""
        all_documents = []
        total_pages = 0
        
        for file in files:
            self._validate_pdf_file(file)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                # Validate page count
                if len(documents) > settings.MAX_PAGES_PER_PDF:
                    raise HTTPException(
                        status_code=400,
                        detail=f"PDF '{file.filename}' has {len(documents)} pages. "
                               f"Maximum allowed is {settings.MAX_PAGES_PER_PDF} pages."
                    )
                
                total_pages += len(documents)
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(documents)
                all_documents.extend(chunks)
                
            finally:
                # Clean up temporary file
                Path(tmp_file_path).unlink(missing_ok=True)
        
        return all_documents, total_pages
    
    async def create_chat_with_documents(
        self,
        files: List[UploadFile],
        model_name: Optional[str] = None
    ) -> ChatCreateResponse:
        """Create a new chat with uploaded PDF documents"""
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one PDF file is required"
            )
        
        # Generate chat ID
        chat_id = self._generate_chat_id()
        
        # Process PDFs
        documents, pages_count = await self._process_pdf_files(files)
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the provided PDFs"
            )
        
        # Use default embedding model if not provided
        embedding_model = model_name or settings.DEFAULT_EMBEDDING_MODEL
        
        # Create vector store
        try:
            self.repository.create_vector_store(
                chat_id=chat_id,
                documents=documents,
                embedding_model_name=embedding_model
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create vector store: {str(e)}"
            )
        
        # Save metadata
        metadata = ChatMetadata(
            chat_id=chat_id,
            created_at=datetime.utcnow(),
            documents_count=len(files),
            chunks_count=len(documents),
            embedding_model=embedding_model,
            last_updated=datetime.utcnow()
        )
        self.repository.save_metadata(chat_id, metadata)
        
        # Save uploaded files
        for file in files:
            await file.seek(0)
            content = await file.read()
            self.repository.save_uploaded_file(chat_id, file.filename, content)
        
        return ChatCreateResponse(
            chat_id=chat_id,
            message="Chat created successfully",
            documents_processed=len(files),
            chunks_created=len(documents)
        )
    
    def check_chat_status(self, chat_id: str) -> ChatStatusResponse:
        """Check if a chat exists and return its status"""
        exists = self.repository.chat_exists(chat_id)
        
        if not exists:
            return ChatStatusResponse(
                exists=False,
                chat_id=chat_id
            )
        
        # Load metadata
        metadata = self.repository.load_metadata(chat_id)
        
        if metadata is None:
            return ChatStatusResponse(
                exists=True,
                chat_id=chat_id
            )
        
        return ChatStatusResponse(
            exists=True,
            chat_id=chat_id,
            documents_count=metadata.documents_count,
            chunks_count=metadata.chunks_count,
            created_at=metadata.created_at
        )
    
    async def add_documents_to_chat(
        self,
        chat_id: str,
        files: List[UploadFile]
    ) -> AddDocumentsResponse:
        """Add new documents to an existing chat"""
        if not self.repository.chat_exists(chat_id):
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} not found"
            )
        
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one PDF file is required"
            )
        
        # Load existing metadata
        metadata = self.repository.load_metadata(chat_id)
        if metadata is None:
            raise HTTPException(
                status_code=500,
                detail=f"Metadata for chat {chat_id} not found"
            )
        
        # Process new PDFs
        new_documents, new_pages = await self._process_pdf_files(files)
        
        if not new_documents:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the provided PDFs"
            )
        
        # Add documents to vector store
        try:
            self.repository.add_documents_to_vector_store(
                chat_id=chat_id,
                new_documents=new_documents,
                embedding_model_name=metadata.embedding_model
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add documents to vector store: {str(e)}"
            )
        
        # Update metadata
        new_total_docs = metadata.documents_count + len(files)
        new_total_chunks = metadata.chunks_count + len(new_documents)
        
        self.repository.update_metadata(
            chat_id=chat_id,
            documents_count=new_total_docs,
            chunks_count=new_total_chunks
        )
        
        # Save uploaded files
        for file in files:
            await file.seek(0)
            content = await file.read()
            self.repository.save_uploaded_file(chat_id, file.filename, content)
        
        return AddDocumentsResponse(
            chat_id=chat_id,
            message="Documents added successfully",
            new_documents_added=len(files),
            new_chunks_added=len(new_documents),
            total_documents=new_total_docs,
            total_chunks=new_total_chunks
        )
    
    def delete_chat(self, chat_id: str) -> DeleteChatResponse:
        """Delete a chat and all associated data"""
        if not self.repository.chat_exists(chat_id):
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} not found"
            )
        
        try:
            deleted = self.repository.delete_chat(chat_id)
            
            return DeleteChatResponse(
                chat_id=chat_id,
                message="Chat deleted successfully" if deleted else "Chat deletion failed",
                deleted=deleted
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete chat: {str(e)}"
            )
    
    def ask_question(
        self,
        chat_id: str,
        prompt: str,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 800
    ) -> AskResponse:
        """Ask a question to the chat using RAG"""
        if not self.repository.chat_exists(chat_id):
            raise HTTPException(
                status_code=404,
                detail=f"Chat {chat_id} not found"
            )
        
        # Load metadata
        metadata = self.repository.load_metadata(chat_id)
        if metadata is None:
            raise HTTPException(
                status_code=500,
                detail=f"Metadata for chat {chat_id} not found"
            )
        
        # Load vector store
        try:
            vector_store = self.repository.load_vector_store(
                chat_id=chat_id,
                embedding_model_name=metadata.embedding_model
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load vector store: {str(e)}"
            )
        
        # Initialize LLM
        model = llm_model_name or settings.DEFAULT_LLM_MODEL
        
        try:
            # Check if we have API keys available
            if settings.OPENAI_API_KEY:
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=settings.OPENAI_API_KEY
                )
            else:
                # Simulation mode - return a mock response
                return AskResponse(
                    chat_id=chat_id,
                    prompt=prompt,
                    answer="[SIMULATION MODE] I would answer your question based on the document context, "
                           "but no LLM API key is configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY "
                           "in your environment variables to enable real responses.",
                    sources_used=settings.RETRIEVAL_K,
                    model_used=f"{model} (simulated)"
                )
            
            # Create prompt template
            prompt_template = """Use the following context to answer the question clearly and precisely.
If you don't have enough information in the context, indicate it clearly.

Context: {context}

Question: {question}

Detailed answer:"""
            
            prompt_obj = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create RAG chain
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": settings.RETRIEVAL_K}
                ),
                chain_type_kwargs={"prompt": prompt_obj}
            )
            
            # Execute query
            result = rag_chain.invoke({"query": prompt})
            answer = result["result"]
            
            return AskResponse(
                chat_id=chat_id,
                prompt=prompt,
                answer=answer,
                sources_used=settings.RETRIEVAL_K,
                model_used=model
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )