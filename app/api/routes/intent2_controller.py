from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models import entities, schemas
from app.models.database import get_db
from app.core.security import get_current_user, has_permission

router = APIRouter()

@router.get("/{intent_id}", response_model=schemas.IntentResponse, tags=["Intent"])
def read_intent(intent_id: int, db: Session = Depends(get_db)):
    """
    Get intent by ID. Users with view permission can access.
    """
    db_intent = db.query(entities.Intent).filter(entities.Intent.intent_id == intent_id).first()
    if db_intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    return db_intent

@router.get("", response_model=List[schemas.IntentResponse2], tags=["Intent"])
def read_intents(db: Session = Depends(get_db)):
    """
    Get a list of intents. Users with view permission can access.
    """
    intents = db.query(entities.Intent).all()
    return intents