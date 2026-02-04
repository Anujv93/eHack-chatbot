def detect_intent(state):
    q = state["query"].lower()

    if any(x in q for x in ["fees", "price", "cost"]):
        intent = "pricing"
    elif any(x in q for x in ["career", "switch", "job"]):
        intent = "career_switch"
    elif any(x in q for x in ["duration", "time"]):
        intent = "duration"
    else:
        intent = "general"

    return {"intent": intent}
