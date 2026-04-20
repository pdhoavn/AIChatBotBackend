from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.schemas import SpecializationResponse
from app.models.entities import Specialization, Major
from typing import List, Optional
from app.core.security import get_current_user

router = APIRouter()

@router.get("", response_model=List[SpecializationResponse])
async def get_all_specializations(
    db: Session = Depends(get_db)
):
    """
    Get all specializations with their articles.
    This endpoint is public and can be accessed without authentication.
    """
    specializations = db.query(Specialization).all()
    
    if not specializations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No specializations found"
        )
    
    return specializations

@router.get("/major/{major_id}", response_model=List[SpecializationResponse])
async def get_specializations_by_major(
    major_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all specializations for a specific major.
    This endpoint is public and can be accessed without authentication.
    """
    # First check if major exists
    major = db.query(Major).filter(Major.major_id == major_id).first()
    if not major:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Major with id {major_id} not found"
        )

    # Get specializations for this major
    specializations = db.query(Specialization).filter(
        Specialization.major_id == major_id
    ).all()
    
    if not specializations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No specializations found for major {major_id}"
        )
    
    return specializations

@router.get("/{specialization_id}", response_model=SpecializationResponse)
async def get_specialization_detail(
    specialization_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific specialization.
    This endpoint is public and can be accessed without authentication.
    """
    specialization = db.query(Specialization).filter(
        Specialization.specialization_id == specialization_id
    ).first()
    
    if not specialization:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Specialization with id {specialization_id} not found"
        )
    
    return specialization