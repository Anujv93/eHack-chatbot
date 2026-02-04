from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict
from rag_service.graph import app as rag_app

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    profile: Dict = Field(default_factory=dict)


@app.post("/chat")
def chat(req: ChatRequest):
    result = rag_app.invoke({
        "query": req.message,
        "history": req.history,
        "profile": req.profile,
    })

    # IMPORTANT: return full state (includes updated profile)
    return result
