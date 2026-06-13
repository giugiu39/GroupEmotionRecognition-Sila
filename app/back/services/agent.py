import os
import httpx
from dotenv import load_dotenv

from schemas import Message

load_dotenv()


async def run_agent(messages: list[Message]) -> str:
    agent_url = os.getenv("AGENT_URL")
    if not agent_url:
        raise RuntimeError("AGENT_URL not configured")

    payload = {
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(agent_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["response"]
    except httpx.ConnectError:
        raise RuntimeError(f"Agent unreachable at {agent_url}")
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Agent returned error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        raise RuntimeError(f"Failed to contact agent: {exc}")
