from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Annotated

from database import get_db
from models import Detection
from schemas import AskAgentRequest, AskAgentResponse
from services.agent import run_agent

router = APIRouter(prefix="/app", tags=["app"])

DbDep = Annotated[AsyncSession, Depends(get_db)]

_EMOTIONS = ["happiness", "neutral", "surprise", "sadness", "fear", "disgust", "contempt", "anger"]


@router.get("/data/getbetweendates")
async def get_between_dates(
    db: DbDep,
    start: int = Query(...),
    end: int = Query(...),
    nodename: str = Query(...),
) -> dict:
    result = await db.execute(
        select(Detection).where(
            Detection.node_name == nodename,
            Detection.timestamp >= start,
            Detection.timestamp <= end,
        )
    )
    detections = result.scalars().all()

    counts: dict[str, int] = {emotion: 0 for emotion in _EMOTIONS}
    for detection in detections:
        if detection.emotion in counts:
            counts[detection.emotion] += 1

    total = len(detections)
    if total == 0:
        percentages = {emotion: 0 for emotion in _EMOTIONS}
    else:
        percentages = {emotion: int(counts[emotion] / total * 100) for emotion in _EMOTIONS}

    return {nodename: percentages}


@router.post("/askagent", response_model=AskAgentResponse)
async def ask_agent(payload: AskAgentRequest) -> AskAgentResponse:
    try:
        reply = await run_agent(payload.messages)
        return AskAgentResponse(response=reply)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
