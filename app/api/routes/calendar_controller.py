from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.utc2_calendar_service import (
    UTC2CalendarError,
    UTC2CalendarService,
)


router = APIRouter()
calendar_service = UTC2CalendarService()


@router.get("/latest")
async def get_latest_calendar():
    """Return the latest public UTC2 working schedule."""
    try:
        result = await calendar_service.answer_query("lịch công tác mới nhất")
        return {
            "title": result.title,
            "post_id": result.post_id,
            "source_url": result.source_url,
            "answer": result.answer,
        }
    except UTC2CalendarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/lookup")
async def lookup_calendar(
    query: str = Query(..., min_length=1),
    today: Optional[date] = Query(
        None,
        description="Override current date for deterministic testing (YYYY-MM-DD).",
    ),
):
    """Resolve a Vietnamese calendar query and return a formatted answer."""
    try:
        result = await calendar_service.answer_query(query, today=today)
        return {
            "title": result.title,
            "post_id": result.post_id,
            "source_url": result.source_url,
            "answer": result.answer,
        }
    except UTC2CalendarError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
