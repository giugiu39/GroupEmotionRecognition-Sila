import os
from dotenv import load_dotenv
import anthropic
from schemas import Message

load_dotenv()

_MODEL = "claude-sonnet-4-20250514"


async def run_agent(messages: list[Message], foto: str | None = None) -> str:
    """Send conversation history and an optional image to the language model and return its reply.

    Raises RuntimeError if the API call fails.
    """
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    api_messages = _build_messages(messages, foto)

    try:
        result = await client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=api_messages,
        )
        return result.content[0].text
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _build_messages(messages: list[Message], foto: str | None) -> list[dict]:
    """Convert internal message objects to the format expected by the messages API.

    If foto is provided it is attached to the last user message as an image block.
    """
    api_messages = []
    for i, msg in enumerate(messages):
        is_last = i == len(messages) - 1
        if is_last and msg.role == "user" and foto is not None:
            content: str | list = _image_blocks(foto) + [{"type": "text", "text": msg.content}]
        else:
            content = msg.content
        api_messages.append({"role": msg.role, "content": content})
    return api_messages


def _image_blocks(foto: str) -> list[dict]:
    """Build an image content block from a base64 string or a URL."""
    if foto.startswith("http://") or foto.startswith("https://"):
        return [{"type": "image", "source": {"type": "url", "url": foto}}]
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": foto,
            },
        }
    ]
