from langchain_openai import ChatOpenAI
from typing import Dict
import json
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_profile_llm(query: str, existing_profile: Dict) -> Dict:
    prompt = f"""
You are extracting student profile information from a chat message.

Current known profile (JSON):
{json.dumps(existing_profile)}

User message:
"{query}"

Extract ONLY what is explicitly or reasonably implied.
If information is not present, return null for that field.

Return STRICT JSON in this format:

{{
  "background": "technical | non-technical | null",
  "career_goal": string | null,
  "qualification": string | null
}}

Rules:
- Be conservative (do not guess wildly)
- Handle typos and informal language
- Do not include explanations
"""

    response = llm.invoke(prompt)

    try:
        data = json.loads(response.content)
    except Exception:
        return existing_profile  # fail-safe

    # Merge with existing profile (do not overwrite known fields)
    merged = existing_profile.copy()
    for k, v in data.items():
        if v and k not in merged:
            merged[k] = v

    return merged
