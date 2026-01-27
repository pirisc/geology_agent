from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    return {"message": "Chat API is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    
    try:
        # Collect all chunks from the generator
        response_text = ""
        for chunk in run_agent(req.message, thread_id):
            response_text += chunk
        
        return {
            "response": response_text,
            "thread_id": thread_id
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
