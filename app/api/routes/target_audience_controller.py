from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models import entities, schemas
from app.models.database import get_db
from app.core.security import get_current_user, has_permission

router = APIRouter()
@router.get("/target-audience", response_model=List[schemas.TargetAudienceSimple])
def get_target_audience(db: Session = Depends(get_db)):
    data = db.query(entities.TargetAudience).all()
    return data