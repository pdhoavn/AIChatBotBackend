from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import schemas, database, entities
from app.core.security import get_current_user

router = APIRouter()

@router.post("/upload", response_model=schemas.AcademicScoreResponse)
def upload_academic_score(
    academic_score: schemas.AcademicScoreCreate,
    db: Session = Depends(database.get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    if not current_user.customer_profile:
        raise HTTPException(status_code=403, detail="User is not a customer")
    
    # Check if academic score already exists for this user
    existing_score = db.query(entities.AcademicScore).filter(
        entities.AcademicScore.customer_id == current_user.user_id
    ).first()
    
    if existing_score:
        # Update existing score
        for key, value in academic_score.model_dump().items():
            setattr(existing_score, key, value)
        db.commit()
        db.refresh(existing_score)
        return existing_score
    else:
        # Create new score
        db_academic_score = entities.AcademicScore(
            **academic_score.model_dump(), 
            customer_id=current_user.user_id
        )
        db.add(db_academic_score)
        db.commit()
        db.refresh(db_academic_score)
        return db_academic_score

@router.get("/users/{user_id}/academic-scores", 
            response_model=schemas.AcademicScoreResponse)
def get_academic_scores(
    user_id: int,
    db: Session = Depends(database.get_db)
):
    # Lấy profile KH (vì bảng academic_score dùng customer_id)
    customer = db.query(entities.CustomerProfile).filter(
        entities.CustomerProfile.customer_id == user_id
    ).first()

    if not customer:
        raise HTTPException(status_code=404, detail="Học sinh không tồn tại")

    scores = db.query(entities.AcademicScore).filter(
        entities.AcademicScore.customer_id == user_id
    ).first()

    return scores
