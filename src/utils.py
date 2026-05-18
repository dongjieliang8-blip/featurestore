import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm_client():
    """Create and return an OpenAI-compatible client configured for MiMo API."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    )
    return client


def call_llm(client, system_prompt, user_message, model=None, temperature=0.3, max_retries=3):
    """Call the LLM with a system prompt and user message, return the response text."""
    if model is None:
        model = os.getenv("DEEPSEEK_MODEL", "mimo-v2.5")
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  [重试 {attempt + 1}/{max_retries}] 等待 {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e


def parse_json_response(text):
    """Parse a JSON response from LLM output, handling markdown code blocks."""
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
