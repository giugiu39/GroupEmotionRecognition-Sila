import httpx
import os
from dotenv import load_dotenv

load_dotenv()

_VLM_URL = os.getenv("VLM_URL", "http://localhost:8001/predict")
_VALID_EMOTIONS = {"happiness", "neutral", "surprise", "distress"}


async def classify_emotion(image_bytes: bytes, filename: str = "image.jpg") -> str | None:
    """Send an image to the VLM server and return the predicted emotion string.

    Returns None if the server is unreachable, times out, or returns an unrecognised value.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            response = await client.post(_VLM_URL, files=files)
            response.raise_for_status()
            data = response.json()
            emotion = data.get("emotion")
            return emotion if emotion in _VALID_EMOTIONS else None
    except Exception:
        return None
