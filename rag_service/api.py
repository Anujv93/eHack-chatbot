from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict
from rag_service.graph import app as rag_app
from datetime import datetime

app = FastAPI()

class LeadRequest(BaseModel):
    name: str
    phone: str



class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    profile: Dict = Field(default_factory=dict)


@app.post("/chat")
def chat(req: ChatRequest):
    result = rag_app.invoke({
        "query": req.message,
        "history": req.history,
    })

    # IMPORTANT: return full state (includes updated profile)
    return result

@app.post("/lead")
def save_lead(lead: LeadRequest):
    with open("leads.txt", "a") as f:
        f.write(
            f"{datetime.now().isoformat()} | {lead.name} | {lead.phone}\n"
        )
    return {"status": "ok"}
