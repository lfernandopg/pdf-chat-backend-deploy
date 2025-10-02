"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class ChatCreateResponse(BaseModel):
    """Response for chat creation"""
    chat_id: str = Field(..., description="Unique identifier for the chat")
    message: str = Field(..., description="Success message")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of text chunks created")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "chat_123abc",
                "message": "Chat created successfully",
                "documents_processed": 2,
                "chunks_created": 45,
                "created_at": "2025-10-02T10:30:00"
            }
        }


class ChatStatusResponse(BaseModel):
    """Response for chat status check"""
    exists: bool = Field(..., description="Whether the chat exists")
    chat_id: str = Field(..., description="Chat identifier")
    documents_count: Optional[int] = Field(None, description="Number of documents in chat")
    chunks_count: Optional[int] = Field(None, description="Number of chunks in vector store")
    created_at: Optional[datetime] = Field(None, description="Chat creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "exists": True,
                "chat_id": "chat_123abc",
                "documents_count": 2,
                "chunks_count": 45,
                "created_at": "2025-10-02T10:30:00"
            }
        }


class AddDocumentsResponse(BaseModel):
    """Response for adding documents to existing chat"""
    chat_id: str = Field(..., description="Chat identifier")
    message: str = Field(..., description="Success message")
    new_documents_added: int = Field(..., description="Number of new documents added")
    new_chunks_added: int = Field(..., description="Number of new chunks added")
    total_documents: int = Field(..., description="Total documents in chat")
    total_chunks: int = Field(..., description="Total chunks in vector store")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "chat_123abc",
                "message": "Documents added successfully",
                "new_documents_added": 1,
                "new_chunks_added": 23,
                "total_documents": 3,
                "total_chunks": 68
            }
        }


class DeleteChatResponse(BaseModel):
    """Response for chat deletion"""
    chat_id: str = Field(..., description="Chat identifier")
    message: str = Field(..., description="Success message")
    deleted: bool = Field(..., description="Whether deletion was successful")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "chat_123abc",
                "message": "Chat deleted successfully",
                "deleted": True
            }
        }


class AskRequest(BaseModel):
    """Request for asking a question to the chat"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="User question")
    llm_model_name: Optional[str] = Field(None, description="LLM model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(800, ge=50, le=4000, description="Maximum tokens to generate")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "¿Cuáles son los puntos principales del documento?",
                "llm_model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 800
            }
        }


class AskResponse(BaseModel):
    """Response for chat question"""
    chat_id: str = Field(..., description="Chat identifier")
    prompt: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources_used: int = Field(..., description="Number of sources used for context")
    model_used: str = Field(..., description="LLM model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "chat_id": "chat_123abc",
                "prompt": "¿Cuáles son los puntos principales?",
                "answer": "Los puntos principales son...",
                "sources_used": 3,
                "model_used": "gpt-3.5-turbo",
                "timestamp": "2025-10-02T10:35:00"
            }
        }


class ErrorResponse(BaseModel):
    """Generic error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input provided",
                "detail": "PDF file exceeds maximum size limit"
            }
        }


class ChatMetadata(BaseModel):
    """Metadata stored with each chat"""
    chat_id: str
    created_at: datetime
    documents_count: int
    chunks_count: int
    embedding_model: str
    last_updated: datetime