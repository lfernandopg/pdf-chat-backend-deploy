"""
API Router for chat endpoints
Handles HTTP requests and delegates to service layer
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from typing import List, Optional

from services.chat_service import ChatService
from repositories.vector_store_repository import VectorStoreRepository
from models.schemas import (
    ChatCreateResponse,
    ChatStatusResponse,
    AddDocumentsResponse,
    DeleteChatResponse,
    AskRequest,
    AskResponse,
    ErrorResponse
)


router = APIRouter()


# Dependency injection
def get_repository() -> VectorStoreRepository:
    """Dependency for vector store repository"""
    return VectorStoreRepository()


def get_chat_service(
    repository: VectorStoreRepository = Depends(get_repository)
) -> ChatService:
    """Dependency for chat service"""
    return ChatService(repository)


@router.post(
    "/upload",
    response_model=ChatCreateResponse,
    status_code=201,
    summary="Upload documents and create a new chat",
    description="Upload one or more PDF files to create a new chat session with RAG capabilities",
    responses={
        201: {"description": "Chat created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def upload_documents_and_create_chat(
    files: List[UploadFile] = File(
        ...,
        description="List of PDF files to upload (max 5 pages each)"
    ),
    model_name: Optional[str] = Query(
        None,
        description="Embedding model to use (defaults to sentence-transformers/all-MiniLM-L6-v2)"
    ),
    service: ChatService = Depends(get_chat_service)
) -> ChatCreateResponse:
    """
    Create a new chat session by uploading PDF documents.
    
    - **files**: One or more PDF files to process
    - **model_name**: Optional embedding model name
    
    Returns the chat_id and processing statistics.
    """
    return await service.create_chat_with_documents(files, model_name)


@router.get(
    "/{chat_id}/status",
    response_model=ChatStatusResponse,
    summary="Check chat status",
    description="Check if a chat exists and retrieve its metadata",
    responses={
        200: {"description": "Chat status retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Chat not found"}
    }
)
def check_chat_exists(
    chat_id: str,
    service: ChatService = Depends(get_chat_service)
) -> ChatStatusResponse:
    """
    Check if a chat exists and get its status.
    
    - **chat_id**: Unique identifier of the chat
    
    Returns existence status and metadata if available.
    """
    return service.check_chat_status(chat_id)


@router.post(
    "/{chat_id}/add-documents",
    response_model=AddDocumentsResponse,
    summary="Add documents to existing chat",
    description="Add new PDF documents to an existing chat session",
    responses={
        200: {"description": "Documents added successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Chat not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
async def add_documents_to_chat(
    chat_id: str,
    files: List[UploadFile] = File(
        ...,
        description="List of PDF files to add to the chat"
    ),
    service: ChatService = Depends(get_chat_service)
) -> AddDocumentsResponse:
    """
    Add new documents to an existing chat.
    
    - **chat_id**: Unique identifier of the chat
    - **files**: One or more PDF files to add
    
    Returns updated statistics including new and total document counts.
    """
    return await service.add_documents_to_chat(chat_id, files)


@router.delete(
    "/{chat_id}",
    response_model=DeleteChatResponse,
    summary="Delete a chat",
    description="Delete a chat and all its associated data (vector store, documents, metadata)",
    responses={
        200: {"description": "Chat deleted successfully"},
        404: {"model": ErrorResponse, "description": "Chat not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def delete_chat(
    chat_id: str,
    service: ChatService = Depends(get_chat_service)
) -> DeleteChatResponse:
    """
    Delete a chat and all associated data.
    
    - **chat_id**: Unique identifier of the chat
    
    Returns deletion confirmation.
    """
    return service.delete_chat(chat_id)


@router.post(
    "/{chat_id}/ask",
    response_model=AskResponse,
    summary="Ask a question to the chat",
    description="Submit a question and get an AI-generated answer based on the chat's documents",
    responses={
        200: {"description": "Answer generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "Chat not found"},
        500: {"model": ErrorResponse, "description": "Server error"}
    }
)
def ask_chat(
    chat_id: str,
    request: AskRequest,
    service: ChatService = Depends(get_chat_service)
) -> AskResponse:
    """
    Ask a question to the chat using RAG.
    
    - **chat_id**: Unique identifier of the chat
    - **request**: Question and LLM parameters
    
    Returns the AI-generated answer with metadata.
    """
    return service.ask_question(
        chat_id=chat_id,
        prompt=request.prompt,
        llm_model_name=request.llm_model_name,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )


@router.get(
    "/",
    summary="List all chats",
    description="Get a list of all available chats (optional future enhancement)"
)
async def list_chats():
    """
    List all available chats.
    
    Note: This is a placeholder for future implementation.
    """
    return {
        "message": "Chat listing not yet implemented",
        "note": "Future enhancement to list all available chat sessions"
    }