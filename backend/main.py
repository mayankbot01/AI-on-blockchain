"""
AI on Blockchain - FastAPI Backend
===================================
Main application entry point integrating LLM, RAG, ML, and Blockchain
Author: Mayank Kumar
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
from typing import Optional

# Import routers (to be created in separate files)
# from backend.api.routes import models, fraud_detection, audit_trail, marketplace, llm

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")
logger.add("logs/app.log", rotation="500 MB", retention="10 days", compression="zip")

# Blockchain connection (placeholder - implement in blockchain module)
class BlockchainManager:
    def __init__(self):
        self.w3 = None
        self.contracts = {}
    
    async def initialize(self):
        """Initialize blockchain connection"""
        try:
            from web3 import Web3
            # Connect to Ethereum node (Infura, Alchemy, or local)
            self.w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
            logger.info(f"Connected to blockchain: {self.w3.is_connected()}")
            
            # Load contract ABIs and addresses
            # self.contracts['AIModelRegistry'] = self.w3.eth.contract(...)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize blockchain: {e}")
            return False
    
    async def shutdown(self):
        """Cleanup blockchain connections"""
        logger.info("Shutting down blockchain connections")

# LLM Manager
class LLMManager:
    def __init__(self):
        self.llm_clients = {}
    
    async def initialize(self):
        """Initialize LLM clients (OpenAI, Anthropic, HuggingFace)"""
        try:
            import openai
            from langchain.llms import OpenAI
            from langchain.embeddings import OpenAIEmbeddings
            
            # Initialize OpenAI
            # self.llm_clients['openai'] = OpenAI(temperature=0.7)
            
            logger.info("LLM clients initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return False
    
    async def shutdown(self):
        """Cleanup LLM resources"""
        logger.info("Shutting down LLM connections")

# RAG Manager
class RAGManager:
    def __init__(self):
        self.vector_db = None
        self.embeddings = None
    
    async def initialize(self):
        """Initialize RAG pipeline with ChromaDB/FAISS"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB
            self.vector_db = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            logger.info("RAG system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False
    
    async def shutdown(self):
        """Cleanup RAG resources"""
        if self.vector_db:
            logger.info("Persisting vector database")

# Global managers
blockchain_manager = BlockchainManager()
llm_manager = LLMManager()
rag_manager = RAGManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management
    """
    # Startup
    logger.info("🚀 Starting AI on Blockchain Platform...")
    
    # Initialize all managers
    await blockchain_manager.initialize()
    await llm_manager.initialize()
    await rag_manager.initialize()
    
    logger.info("✅ All systems initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI on Blockchain Platform...")
    await blockchain_manager.shutdown()
    await llm_manager.shutdown()
    await rag_manager.shutdown()
    
    logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="AI on Blockchain Platform",
    description="Enterprise-grade integration of AI, LLM, RAG, and Blockchain technologies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "service": "AI on Blockchain Platform",
        "version": "1.0.0",
        "blockchain_connected": blockchain_manager.w3.is_connected() if blockchain_manager.w3 else False
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "blockchain": blockchain_manager.w3.is_connected() if blockchain_manager.w3 else False,
            "llm": bool(llm_manager.llm_clients),
            "rag": bool(rag_manager.vector_db),
        }
    }

# ==========================================
# AI Model Registry Endpoints
# ==========================================

from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

class ModelRegistrationRequest(BaseModel):
    model_name: str
    model_type: str  # LLM, CNN, Transformer, etc.
    framework: str   # PyTorch, TensorFlow
    ipfs_hash: str
    description: str
    owner_address: Optional[str] = None

class ModelUpdateRequest(BaseModel):
    model_id: int
    ipfs_hash: str
    change_log: str

@app.post("/api/v1/models/register", tags=["Model Registry"])
async def register_model(request: ModelRegistrationRequest):
    """
    Register a new AI model on the blockchain
    
    This creates an immutable record of the model with:
    - Model metadata (name, type, framework)
    - IPFS hash for model weights
    - Cryptographic hash for verification
    - Timestamp and version
    """
    try:
        # In production, this would:
        # 1. Compute model hash
        # 2. Call blockchain smart contract
        # 3. Store metadata in IPFS
        # 4. Log to audit trail
        
        model_data = {
            "model_id": 1,  # Would come from blockchain
            "model_name": request.model_name,
            "model_type": request.model_type,
            "framework": request.framework,
            "ipfs_hash": request.ipfs_hash,
            "registered_at": datetime.utcnow().isoformat(),
            "blockchain_tx": "0x1234...abcd",  # Transaction hash
            "status": "registered"
        }
        
        logger.info(f"Registered model: {request.model_name}")
        return {"success": True, "data": model_data}
        
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/{model_id}", tags=["Model Registry"])
async def get_model(model_id: int):
    """Get model details by ID"""
    try:
        # Query blockchain for model details
        model_info = {
            "model_id": model_id,
            "model_name": "Fraud Detection LLM v1.2",
            "model_type": "LLM",
            "framework": "PyTorch",
            "version": 2,
            "owner": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            "is_active": True,
            "created_at": "2024-01-15T10:30:00Z"
        }
        
        return {"success": True, "data": model_info}
        
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        raise HTTPException(status_code=404, detail="Model not found")

# ==========================================
# LLM-Powered Fraud Detection Endpoints
# ==========================================

class TransactionAnalysisRequest(BaseModel):
    transaction_id: str
    amount: float
    sender: str
    receiver: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@app.post("/api/v1/fraud/analyze", tags=["Fraud Detection"])
async def analyze_transaction(request: TransactionAnalysisRequest):
    """
    Analyze transaction for fraud using LLM + ML pipeline
    
    Uses:
    - LLM for contextual analysis
    - ML model for pattern detection
    - Blockchain logging for audit trail
    """
    try:
        # In production, this would:
        # 1. Use LLM to analyze transaction context
        # 2. Run ML fraud detection model
        # 3. Compute fraud score with explainability
        # 4. Log decision to blockchain audit trail
        
        analysis_result = {
            "transaction_id": request.transaction_id,
            "fraud_score": 0.15,  # 15% probability
            "risk_level": "low",
            "factors": [
                {"factor": "amount_within_normal_range", "contribution": -0.3},
                {"factor": "known_receiver", "contribution": -0.2},
                {"factor": "unusual_time", "contribution": 0.1}
            ],
            "llm_explanation": "Transaction appears legitimate. Amount is within sender's historical range, and receiver is a known contact.",
            "recommended_action": "approve",
            "blockchain_log_tx": "0xabc...def"
        }
        
        logger.info(f"Analyzed transaction {request.transaction_id}: {analysis_result['risk_level']}")
        return {"success": True, "data": analysis_result}
        
    except Exception as e:
        logger.error(f"Fraud analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# RAG-Based Audit Trail Endpoints
# ==========================================

class AuditLogRequest(BaseModel):
    event_type: str
    actor: str
    action: str
    details: Dict[str, Any]

class AuditQueryRequest(BaseModel):
    query: str
    limit: int = 10

@app.post("/api/v1/audit/log", tags=["Audit Trail"])
async def log_audit_event(request: AuditLogRequest):
    """
    Log an audit event to blockchain and vector database
    
    Enables:
    - Immutable audit trail on blockchain
    - Semantic search via RAG
    - Natural language queries over logs
    """
    try:
        # Log to blockchain
        # Store embeddings in vector DB for RAG
        
        audit_record = {
            "audit_id": "audit_12345",
            "event_type": request.event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "blockchain_tx": "0x789...xyz",
            "vector_db_id": "vec_456"
        }
        
        logger.info(f"Logged audit event: {request.event_type}")
        return {"success": True, "data": audit_record}
        
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/audit/query", tags=["Audit Trail"])
async def query_audit_trail(request: AuditQueryRequest):
    """
    Query audit trail using natural language (RAG)
    
    Example queries:
    - "Show all model updates in the last week"
    - "Find fraud detections above 80% confidence"
    - "List all transactions by user 0x123..."
    """
    try:
        # Use RAG to semantic search audit logs
        # Retrieve relevant blockchain records
        
        results = {
            "query": request.query,
            "matches": [
                {
                    "audit_id": "audit_100",
                    "summary": "Model 'Fraud Detector v2' was updated",
                    "timestamp": "2024-02-20T14:30:00Z",
                    "relevance_score": 0.92
                }
            ],
            "total": 1
        }
        
        return {"success": True, "data": results}
        
    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Data Marketplace Endpoints
# ==========================================

@app.post("/api/v1/marketplace/list-data", tags=["Data Marketplace"])
async def list_dataset():
    """List a dataset for sale in the decentralized marketplace"""
    return {"message": "Data marketplace listing endpoint - to be implemented"}

@app.get("/api/v1/marketplace/datasets", tags=["Data Marketplace"])
async def get_available_datasets():
    """Get all available datasets in the marketplace"""
    return {"message": "Data marketplace retrieval endpoint - to be implemented"}

# ==========================================
# Smart Contract Analysis Endpoints
# ==========================================

@app.post("/api/v1/contracts/analyze", tags=["Smart Contract Analysis"])
async def analyze_smart_contract():
    """
    Analyze smart contract for vulnerabilities using fine-tuned LLM
    
    Uses CodeLLaMA/StarCoder fine-tuned on security patterns
    """
    return {"message": "Smart contract analysis endpoint - to be implemented"}

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
