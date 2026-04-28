from fastapi import APIRouter, Depends, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models import entities, schemas
from app.models.database import get_db
from app.core.security import get_current_user, has_permission
from app.services.training_service import TrainingService

router = APIRouter()


@router.get("/suggestions", response_model=List[schemas.SuggestionQuestionResponse])
def get_suggestions(
    target_audience_id: int = Query(...),
    intent_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    return TrainingService.get_suggestion_questions(db, target_audience_id, intent_id)
