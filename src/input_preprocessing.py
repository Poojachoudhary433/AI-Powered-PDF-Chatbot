import json
from typing import Dict, Any
from model import query_ollama

# ---- Prompt Templates ----
CHITCHAT_PROMPT = "You are a friendly assistant. Respond conversationally.all_chat_text :{all_chat_text}, all_ocr_text : {all_ocr_text}"
bill_parsing_PROMPT = """
You are a helpful assistant that extracts bill details.

### Task
Analyze the following OCR text of a bill and return a JSON object strictly following this schema:
{{
  "header": [{{"key": "string", "value": "string"}}],
  "seller_info": [{{"key": "string", "value": "string"}}],
  "buyer_info": [{{"key": "string", "value": "string"}}],
  "totals": [{{"key": "string", "value": "string"}}],
  "line_items": [{{"description": "string", "quantity": "string", "unit_price": "string", "total": "string"}}]
}}

### Rules
1. Return only valid JSON (no extra text, explanation, or markdown).
2. Use empty lists for missing sections.
3. All values must be strings.

### OCR Text
{all_ocr_text}
### all_chat_text : {all_chat_text}
"""


# ---- Intent Detection ----
## Here this is manual detection -- so for input in certain format like for export or bill parsing
## need to include if "export" in normalized or "download" in normalized: or "invoice" in normalized or "bill" in normalized or "parse" in normalized
def detect_intent(user_input: str) -> str:
    """
    Simple rule-based detection of intent.
    Returns one of: chit_chat, bill_parsing, data_export, info_query
    """
    normalized = user_input.lower()
    if "export" in normalized or "download" in normalized:
        return "data_export"
    elif "invoice" in normalized or "bill" in normalized or "parse" in normalized:
        return "bill_parsing"
    else:
        return "chit_chat"

# ---- Main Processing ----
def process_input(user_input: str)-> Dict[str, Any]:
    """
    Processes input based on intent and calls appropriate model functions.
    bill_data: previously extracted bill JSON (if available) for info_query
    """
    intent = detect_intent(user_input)

    if intent == "chit_chat":
        return {"intent": intent, "prompt": CHITCHAT_PROMPT}

    elif intent == "bill_parsing":
        return {"intent": intent, "prompt": bill_parsing_PROMPT}

    elif intent == "data_export":
        return {"intent": intent, "prompt": bill_parsing_PROMPT}

    else:
        return {"intent": intent, "prompt": CHITCHAT_PROMPT}


if __name__ == '__main__':
    pass