from .profile_extractor import extract_profile_llm


def update_profile_from_query(state):
    query = state["query"]
    existing_profile = state.get("profile", {})

    updated_profile = extract_profile_llm(query, existing_profile)

    return {"profile": updated_profile}


def should_ask_profile_question(state):
    profile = state.get("profile", {})

    if "career_goal" not in profile:
        return {
            "reply": "What kind of career are you aiming for?",
            "ask_profile": True,
        }

    if "background" not in profile:
        return {
            "reply": "Do you come from a technical or non-technical background?",
            "ask_profile": True,
        }

    return {"ask_profile": False}
