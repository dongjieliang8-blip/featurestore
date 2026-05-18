import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    )


def call_llm(client, system_prompt, user_message, model=None, temperature=0.3):
    if model is None:
        model = os.getenv("DEEPSEEK_MODEL", "mimo-v2.5")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
        max_tokens=8192
    )
    return response.choices[0].message.content


def parse_json_response(text):
    if not text or not text.strip():
        return {"raw_response": text, "error": "empty_response"}
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": text, "error": "json_parse_failed"}
