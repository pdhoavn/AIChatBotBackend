from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models import entities, schemas
from app.models.database import get_db
from app.core.security import get_current_user, has_permission

router = APIRouter()

def check_create_edit_permission(current_user: entities.Users = Depends(get_current_user)):
    if not current_user or not (has_permission(current_user, "admin") or has_permission(current_user, "consultant")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Consultant permission required"
        )
    return current_user

def check_view_permission(current_user: entities.Users = Depends(get_current_user)):
    # 1. Kiểm tra user tồn tại
    if not current_user:
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated")

    try:
        user_perms_list = [p.permission_name.lower() for p in current_user.permissions] 
    except AttributeError:
        user_perms_list = [p.lower() for p in current_user.permissions]

    is_admin_or_consultant = "admin" in user_perms_list or "consultant" in user_perms_list

    is_admission_related = any("admission" in p for p in user_perms_list)

    if not (is_admin_or_consultant or is_admission_related):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin, Consultant, or Admission permission required"
        )
    
    return current_user

@router.post("", response_model=schemas.IntentResponse, tags=["Intent"])
def create_intent(intent: schemas.IntentBase, db: Session = Depends(get_db), current_user: entities.Users = Depends(check_create_edit_permission)):
    """
    Post intent for admin or consultant only
    """
    db_intent = entities.Intent(
        intent_name=intent.intent_name,
        description=intent.description,
        created_by=current_user.user_id
    )
    db.add(db_intent)
    db.commit()
    db.refresh(db_intent)
    return db_intent

@router.get("/active", response_model=List[schemas.IntentResponse], tags=["Intent"])
def read_active_intents(db: Session = Depends(get_db), current_user: entities.Users = Depends(check_view_permission)):
    """
    Get a list of active intents. Users with view permission can access.
    """
    intents = db.query(entities.Intent).filter(entities.Intent.is_deleted == False).all()
    return intents

@router.get("/{intent_id}", response_model=schemas.IntentResponse, tags=["Intent"])
def read_intent(intent_id: int, db: Session = Depends(get_db), current_user: entities.Users = Depends(check_view_permission)):
    """
    Get intent by ID. Users with view permission can access.
    """
    db_intent = db.query(entities.Intent).filter(entities.Intent.intent_id == intent_id).first()
    if db_intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    return db_intent


@router.get("", response_model=List[schemas.IntentResponse], tags=["Intent"])
def read_intents(db: Session = Depends(get_db), current_user: entities.Users = Depends(check_view_permission)):
    """
    Get a list of intents. Users with view permission can access.
    """
    intents = db.query(entities.Intent).all()
    return intents


@router.put("/{intent_id}", response_model=schemas.IntentResponse, tags=["Intent"])
def update_intent(intent_id: int, intent: schemas.IntentBase, db: Session = Depends(get_db), current_user: entities.Users = Depends(check_create_edit_permission)):
    """
    Update and restore intent by ID. Admin or Consultant permission required.
    """
    db_intent = db.query(entities.Intent).filter(entities.Intent.intent_id == intent_id).first()
    if db_intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    
    db_intent.intent_name = intent.intent_name
    db_intent.description = intent.description
    db_intent.is_deleted = False
    
    db.commit()
    db.refresh(db_intent)
    return db_intent


@router.delete("/{intent_id}", tags=["Intent"])
def delete_intent(intent_id: int, db: Session = Depends(get_db), current_user: entities.Users = Depends(check_create_edit_permission)):
    """
    Soft delete intent by ID. Admin or Consultant permission required.
    """
    db_intent = db.query(entities.Intent).filter(entities.Intent.intent_id == intent_id, entities.Intent.is_deleted == False).first()
    if db_intent is None:
        raise HTTPException(status_code=404, detail="Intent not found")
    
    db_intent.is_deleted = True
    
    db.commit()
    return {"message": "Intent deleted successfully"}