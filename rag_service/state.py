from typing import TypedDict, List, Optional

class LeadProfile(TypedDict, total=False):
    qualification: str
    background: str
    career_goal: str
    experience_years: int

class ChatState(TypedDict):
    query: str
    history: List[dict]

    profile: LeadProfile
    context: str

    ask_profile: bool
    ask_lead: bool

    reply: str
    suggestions: List[str]
