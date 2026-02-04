def recommend_course(state):
    intent = state["intent"]
    lead = state.get("lead", {})

    course = None

    if intent == "career_switch":
        course = "Full Stack Web Development"
    elif intent == "general":
        course = "Foundation Programming"

    return {"recommended_course": course}
