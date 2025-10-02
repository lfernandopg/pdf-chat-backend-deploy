"""
FastAPI RAG Application - Main Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path

#from app.api import chats
from src.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup: Create necessary directories
    Path(settings.STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: Cleanup if needed
    pass


# Initialize FastAPI application
app = FastAPI(
    title="RAG PDF Chat API",
    description="Sistema de RAG sobre documentos PDF con arquitectura escalable",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
#app.include_router(chats.router, prefix="/api/v1/chats", tags=["chats"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "PDF Chat API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "storage_path": str(settings.STORAGE_PATH),
        "storage_exists": Path(settings.STORAGE_PATH).exists()
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )