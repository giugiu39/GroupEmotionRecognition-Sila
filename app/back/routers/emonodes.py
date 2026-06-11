from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from database import get_db
from models import Detection
from services.vlm import classify_emotion

router = APIRouter(prefix="/emonodes", tags=["emonodes"])

DbDep = Annotated[AsyncSession, Depends(get_db)]


@router.post("/sendmessage", status_code=200)
async def send_message(
    db: DbDep,
    foto: UploadFile = File(...),
    node_name: str = Form(...),
    num_persone: int = Form(...),
    timestamp: int = Form(...),
) -> Response:
    """Receive a frame from a Raspberry Pi node, store it, and update it with the VLM emotion result."""
    image_bytes = await foto.read()

    detection = Detection(
        node_name=node_name,
        num_persone=num_persone,
        emotion=None,
        timestamp=timestamp,
    )
    db.add(detection)
    await db.commit()
    await db.refresh(detection)

    emotion = await classify_emotion(image_bytes, foto.filename or "image.jpg")

    if emotion is not None:
        detection.emotion = emotion
        await db.commit()

    return Response(status_code=200)
