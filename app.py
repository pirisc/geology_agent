from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your agent
from agent import run_agent

app = FastAPI(
    title="Geology Chat API",
    description="AI-powered geology assistant with expert knowledge",
    version="1.0.0"
)

# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    thread_id: str | None = Field(None, description="Optional conversation thread ID")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or only whitespace')
        return v.strip()

# ═══════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def read_root():
    return {
        "message": "Geology Chat API is running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """
    Stream chat responses from the geology assistant.
    
    Returns Server-Sent Events (SSE) stream with:
    - thread_id: Conversation identifier
    - content: Streamed response tokens
    - [DONE]: Stream completion signal
    """
    # Rate limiting
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    thread_id = req.thread_id or str(uuid.uuid4())
    
    logger.info(f"Chat request - Thread: {thread_id}, IP: {client_ip}")
    
    try:
        async def generate():
            try:
                # Send thread_id first
                yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"
                
                # Stream tokens with timeout protection
                token_count = 0
                async for chunk in run_agent(req.message, thread_id):
                    token_count += 1
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                    
                    # Prevent infinite loops
                    if token_count > 50000:
                        logger.warning(f"Token limit reached for thread {thread_id}")
                        break
                
                # Signal completion
                yield "data: [DONE]\n\n"
                logger.info(f"Chat completed - Thread: {thread_id}, Tokens: {token_count}")
                
            except asyncio.CancelledError:
                logger.info(f"Stream cancelled by client - Thread: {thread_id}")
                raise
            except Exception as e:
                logger.error(f"Error in generate() - Thread: {thread_id}, Error: {str(e)}")
                error_msg = f"data: {json.dumps({'error': 'An error occurred during response generation'})}\n\n"
                yield error_msg
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in nginx
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint - Thread: {thread_id}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
