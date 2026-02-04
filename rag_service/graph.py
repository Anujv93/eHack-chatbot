from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from .state import ChatState
from .retriever import load_retriever
from .profile import update_profile_from_query, should_ask_profile_question
from .lead import should_collect_lead
from .lead import handle_lead_capture



load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
retriever = load_retriever()


# -------------------------
# Retrieval (PROFILE-AWARE)
# -------------------------
def retrieve(state: ChatState):
    profile = state.get("profile", {})
    query = state["query"]

    enriched_query = f"""
User background: {profile.get('background')}
Qualification: {profile.get('qualification')}
Career goal: {profile.get('career_goal')}

Question:
{query}
"""

    docs = retriever.invoke(enriched_query)
    context = "\n\n".join(d.page_content for d in docs)

    return {"context": context}


# -------------------------
# Answer Generation
# -------------------------
def answer(state: ChatState):
    profile = state.get("profile", {})
    context = state.get("context", "")

    prompt = f"""
You are a friendly admission counsellor.

Student profile:
- Qualification: {profile.get('qualification')}
- Background: {profile.get('background')}
- Goal: {profile.get('career_goal')}

RULES:
- Recommend ONE best option
- Keep response short and scannable
- Avoid listing everything
- Sound human and supportive

Context:
{context}

Student message:
{state['query']}
"""

    response = llm.invoke(prompt)

    return {
        "reply": response.content,
        "suggestions": [
            "View syllabus",
            "Check fees",
            "Talk to counsellor",
        ],
    }

# -------------------------
# Build Graph
# -------------------------
graph = StateGraph(ChatState)

graph.add_node("profile_update", update_profile_from_query)
graph.add_node("profile_question", should_ask_profile_question)
graph.add_node("retrieve", retrieve)
graph.add_node("lead_check", should_collect_lead)
graph.add_node("lead_capture", handle_lead_capture)
graph.add_node("answer", answer)

# Entry point
graph.set_entry_point("profile_update")

# Profile understanding
graph.add_edge("profile_update", "profile_question")

# If we need more profile info, stop and wait for user
graph.add_conditional_edges(
    "profile_question",
    lambda s: s.get("ask_profile"),
    {
        True: END,
        False: "retrieve",
    },
)

# Retrieval → lead decision
graph.add_edge("retrieve", "lead_check")

# Decide whether to collect lead
graph.add_conditional_edges(
    "lead_check",
    lambda s: s.get("ask_lead"),
    {
        True: "lead_capture",
        False: "answer",
    },
)

# End states
graph.add_edge("lead_capture", END)
graph.add_edge("answer", END)

# Compile
app = graph.compile()

