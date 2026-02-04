def update_profile_from_query(state):
    query = state["query"].lower()
    profile = state.get("profile", {})

    # Infer qualification
    if "btech" in query or "b.tech" in query:
        profile["qualification"] = "B.Tech"
    elif "12th" in query:
        profile["qualification"] = "12th pass"
    elif "graduate" in query:
        profile["qualification"] = "Graduate"

    # Infer background
    if "non tech" in query or "non-technical" in query:
        profile["background"] = "non-technical"
    elif "developer" in query or "it" in query:
        profile["background"] = "technical"

    # Infer career goal
    if any(x in query for x in ["job", "career", "switch"]):
        profile["career_goal"] = query

    return {"profile": profile}


def should_ask_profile_question(state):
    profile = state.get("profile", {})

    if "career_goal" not in profile:
        return {
            "reply": "Before I suggest anything, what kind of career are you aiming for?",
            "ask_profile": True,
        }

    if "background" not in profile:
        return {
            "reply": "Do you come from a technical or non-technical background?",
            "ask_profile": True,
        }

    return {"ask_profile": False}
