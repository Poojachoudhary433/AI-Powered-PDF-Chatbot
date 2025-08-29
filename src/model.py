import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
import json
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
# ---- Thread pool for blocking calls ----
executor = ThreadPoolExecutor(max_workers=2)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


import json

def enforce_bill_schema(response_text: str) -> dict:
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # if LLM returns invalid JSON, wrap it as raw text
        return {
            "header": [],
            "seller_info": [],
            "buyer_info": [],
            "totals": [],
            "line_items": [],
            "raw_text": response_text
        }

    # Ensure all keys exist
    for key in ["header", "seller_info", "buyer_info", "totals", "line_items"]:
        data.setdefault(key, [])

    return data



# ---- Blocking Ollama call (kept for general queries) ----
def run_ollama_sync(model_name: str, prompt: str):
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        capture_output=True,
        check=True
    )
    return result.stdout.decode("utf-8").strip()

async def run_ollama_async(model_name: str, prompt: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, run_ollama_sync, model_name, prompt)

# ---- Model Specific Wrappers ----
async def query_llama3(prompt: str):
    return await run_ollama_async("llama3", prompt)

async def query_phi(prompt: str):
    return await run_ollama_async("phi", prompt)

async def query_mistral(prompt: str):
    return await run_ollama_async("mistral", prompt)

# ---- OpenAI Bill Parsing ----
def query_openai_bill(prompt: str) -> str:
    bill_prompt = prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": bill_prompt}],
        temperature=0
    )
    raw_content = response.choices[0].message.content
    print(f"raw_content : {raw_content}")
    return enforce_bill_schema(raw_content)

# ---- Unified entry ----
async def query_ollama(prompt: str, model_choice: str = "llama3", enforce_bill_schema: bool = False, bill_mode: bool = False):
    """
    Unified async entry.
    - If bill_mode=True, uses OpenAI to extract bill details in structured JSON.
    - If enforce_bill_schema=True (for Ollama), post-processes response to wrap schema.
    """
    if bill_mode:
        return query_openai_bill(prompt)

    if model_choice == "phi":
        response = await query_phi(prompt)
    elif model_choice == "mistral":
        response = await query_mistral(prompt)
    else:
        response = await query_llama3(prompt)

    if enforce_bill_schema:
        return json.dumps({
            "header": [],
            "seller_info": [],
            "buyer_info": [],
            "totals": [],
            "line_items": [],
            "raw_text": response
        }, indent=2)
    return response


if __name__ == '__main__':
    pass