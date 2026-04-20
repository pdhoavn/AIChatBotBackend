from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.schemas import MajorDetailResponse
from app.models.entities import Major, Users
from typing import List, Optional
from app.core.security import get_current_user

router = APIRouter()

@router.get("", response_model=List[MajorDetailResponse])
async def get_all_majors(
    db: Session = Depends(get_db),
    current_user: Optional[Users] = Depends(get_current_user)
):
    """
    Get all majors with their curriculum and courses information.
    This endpoint is public and can be accessed without authentication.
    """
    majors = db.query(Major).all()
    
    if not majors:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No majors found"
        )
    
    # Format response
    response = []
    for major in majors:
        major_data = {
            "major_id": major.major_id,
            "major_name": major.major_name,
            # Add relations
            "articles": [{
                "article_id": article.article_id,
                "title": article.title,
                "description": article.description,
                "url": article.url,
                "create_at": article.create_at,
                "specialization": {
                    "specialization_id": article.specialization.specialization_id,
                    "specialization_name": article.specialization.specialization_name
                } if article.specialization else None
            } for article in major.articles],
            "admission_forms": [{
                "form_id": form.form_id,
                "fullname": form.fullname,
                "email": form.email,
                "phone_number": form.phone_number,
                "campus": form.campus,
                "submit_time": form.submit_time
            } for form in major.admission_forms]
        }
        response.append(major_data)

    return response

@router.get("/{major_id}", response_model=MajorDetailResponse)
async def get_major_detail(
    major_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[Users] = Depends(get_current_user)
):
    """
    Get detailed information about a specific major.
    This endpoint is public and can be accessed without authentication.
    """
    major = db.query(Major).filter(Major.major_id == major_id).first()
    
    if not major:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Major with id {major_id} not found"
        )
    
    # Format response
    response = {
        "major_id": major.major_id,
        "major_name": major.major_name,
        # Add relations
        "articles": [{
            "article_id": article.article_id,
            "title": article.title,
            "description": article.description,
            "url": article.url,
            "create_at": article.create_at,
            "specialization": {
                "specialization_id": article.specialization.specialization_id,
                "specialization_name": article.specialization.specialization_name
            } if article.specialization else None
        } for article in major.articles],
        "admission_forms": [{
            "form_id": form.form_id,
            "fullname": form.fullname,
            "email": form.email,
            "phone_number": form.phone_number,
            "campus": form.campus,
            "submit_time": form.submit_time
        } for form in major.admission_forms]
    }

    return response