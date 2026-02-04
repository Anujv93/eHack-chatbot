from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from rag_service.graph import app as rag_app

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    profile: Dict = {}

@app.post("/chat")
def chat(req: ChatRequest):
    result = rag_app.invoke({
        "query": req.message,
        "history": req.history,
        "profile": req.profile,
    })
    return result
