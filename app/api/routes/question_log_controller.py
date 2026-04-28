from fastapi import APIRouter, Depends, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models import entities, schemas
from app.models.database import get_db
from app.core.security import get_current_user, has_permission
from app.services.training_service import TrainingService

router = APIRouter()


@router.get(
    "/suggestions",
    response_model=List[schemas.SuggestionTrainingResponse],
)
def get_suggestions_from_training_api(
    target_audience_id: int = Query(...),
    intent_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    return TrainingService.get_suggestion_from_training(
        db=db, target_audience_id=target_audience_id, intent_id=intent_id
    )
