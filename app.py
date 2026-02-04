from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from agent import run_agent
import json

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Chat API is running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    
    try:
        async def generate():
            # Send thread_id first - properly formatted JSON
            yield f"data: {json.dumps({'thread_id': thread_id})}\n\n"
            
            # Stream tokens
            async for chunk in run_agent(req.message, thread_id):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # Signal completion
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
