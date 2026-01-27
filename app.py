from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from agent import run_agent
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    
    async def event_stream():
        yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'start'})}\n\n"
        
        for chunk in run_agent(req.message, thread_id):
            yield f"data: {json.dumps({'chunk': chunk, 'type': 'content'})}\n\n"
        
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
