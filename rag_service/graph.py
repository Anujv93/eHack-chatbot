from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .state import ChatState
from .retriever import load_retriever

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
retriever = load_retriever()


# -------------------------
# Retrieval
# -------------------------
def retrieve(state: ChatState):
    query = state["query"]

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    return {"context": context}


# -------------------------
# Answer
# -------------------------
def answer(state: ChatState):
    prompt = f"""
You are a helpful course assistant for eHack Academy.

Rules:
- Be concise and clear
- Structure answers nicely
- Do not hard-sell
- Use bullets where helpful

Context:
{state.get("context", "")}

User question:
{state["query"]}
"""

    response = llm.invoke(prompt)

    return {
        "reply": response.content
    }


# -------------------------
# Build Graph (SIMPLE)
# -------------------------
graph = StateGraph(ChatState)

graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

app = graph.compile()
