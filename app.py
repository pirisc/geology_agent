from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from agent import run_agent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@app.get("/")
def read_root():
    return {"message": "Chat API is running. POST to /chat to interact."}    

@app.post("/chat")
def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    response = run_agent(req.message, thread_id)

    return {
        "thread_id": thread_id,
        "response": response
    }
