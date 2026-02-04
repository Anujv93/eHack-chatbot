def should_ask_lead(state):
    intent = state["intent"]
    history_len = len(state.get("history", []))

    ask = False

    if intent in ["pricing", "career_switch"] and history_len >= 2:
        ask = True

    return {"ask_lead": ask}
