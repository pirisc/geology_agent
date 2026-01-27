from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
import json
from agent import run_agent
from fastapi.middleware.cors import CORSMiddleware
import asyncio

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
    return {"message": "Chat API is running. POST to /chat to interact."}

@app.options("/chat")
def options_chat():
    return {}

@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    
    async def event_stream():
        try:
            # Send thread_id first
            yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'start'})}\n\n"
            
            # Run the agent in an executor since it's sync
            loop = asyncio.get_event_loop()
            
            # Collect chunks from the generator
            def run_sync_agent():
                chunks = []
                for chunk in run_agent(req.message, thread_id):
                    chunks.append(chunk)
                return chunks
            
            chunks = await loop.run_in_executor(None, run_sync_agent)
            
            # Stream the chunks
            for chunk in chunks:
                yield f"data: {json.dumps({'chunk': chunk, 'type': 'content'})}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
            
            # Send end signal
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        except Exception as e:
            print(f"Error in stream: {e}")
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
