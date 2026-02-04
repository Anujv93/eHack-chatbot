import re
from .save_lead import save_lead_to_file


def should_collect_lead(state):
    profile = state.get("profile", {})
    history_len = len(state.get("history", []))

    # Only ask for lead when user is engaged and goal is known
    if history_len >= 4 and "career_goal" in profile:
        return {"ask_lead": True}

    return {"ask_lead": False}

def extract_name_phone(text: str):
    phone_match = re.search(r"\b[6-9]\d{9}\b", text)
    name_match = re.search(r"(my name is|i am)\s+([a-zA-Z ]+)", text, re.I)

    phone = phone_match.group() if phone_match else None
    name = name_match.group(2).strip() if name_match else None

    return name, phone


def handle_lead_capture(state):
    query = state["query"]
    profile = state.get("profile", {})
    history = state.get("history", [])

    name, phone = extract_name_phone(query)

    if name and phone:
        save_lead_to_file(
            name=name,
            phone=phone,
            profile=profile,
            history=history,
        )
        return {
            "reply": "Thanks! 😊 Our counsellor will contact you shortly.",
            "ask_lead": False,
        }

    return {
        "reply": (
            "Could you please share your **name and phone number**?\n"
            "Example: My name is Rahul, my number is 9876543210"
        ),
        "ask_lead": True,
    }
