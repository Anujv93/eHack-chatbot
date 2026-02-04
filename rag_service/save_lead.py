import os
from datetime import datetime
from typing import Dict, List

BASE_DIR = os.path.dirname(__file__)
LEADS_FILE = os.path.join(BASE_DIR, "storage", "leads.txt")

# Ensure storage folder exists
os.makedirs(os.path.dirname(LEADS_FILE), exist_ok=True)


def save_lead_to_file(
    name: str,
    phone: str,
    profile: Dict,
    history: List[dict],
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    last_messages = history[-6:]  # keep last few turns only

    with open(LEADS_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Name: {name}\n")
        f.write(f"Phone: {phone}\n\n")

        f.write("Profile:\n")
        for k, v in profile.items():
            f.write(f"  - {k}: {v}\n")

        f.write("\nConversation:\n")
        for msg in last_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            f.write(f"{role.upper()}: {content}\n")

        f.write("=" * 60 + "\n")
