from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.models import entities, schemas
from app.models.database import get_db
from app.core.security import get_current_user, has_permission

router = APIRouter()

# =================================================================
# TEMPLATE ROUTES
# =================================================================

@router.post("", response_model=schemas.TemplateResponse, status_code=status.HTTP_201_CREATED)
def create_template(
    template: schemas.TemplateCreate,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    """
    Create template. Admin permission required.
    """
    if not has_permission(current_user, "Admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to create templates"
        )
    
    db_template = entities.Template(
        template_name=template.template_name,
        description=template.description,
        created_by=current_user.user_id,
        is_active=True
    )
    db.add(db_template)
    db.flush()  # Use flush to get the template_id before committing

    for qa_data in template.qa_pairs:
        db_qa = entities.Template_QA(
            **qa_data.dict(),
            template_id=db_template.template_id
        )
        db.add(db_qa)
    
    db.commit()
    db.refresh(db_template)
    return db_template

@router.get("", response_model=List[schemas.TemplateResponse])
def read_templates(db: Session = Depends(get_db), current_user: entities.Users = Depends(get_current_user)):
    """
    Read templates. Admin or Consultant permission required.
    """
    if has_permission(current_user, "Admin"):
        templates = db.query(entities.Template).filter(entities.Template.is_active == True).all()
    elif has_permission(current_user, "Consultant"):
        templates = db.query(entities.Template).filter(entities.Template.is_active == True).all()
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view templates"
        )
    return templates

@router.get("/{template_id}", response_model=schemas.TemplateResponse)
def read_template(template_id: int, db: Session = Depends(get_db), current_user: entities.Users = Depends(get_current_user)):
    """
    Read template by ID. Admin or Consultant permission required.
    """
    db_template = db.query(entities.Template).filter(entities.Template.template_id == template_id).first()
    if db_template is None:
        raise HTTPException(status_code=404, detail="Template not found")

    if has_permission(current_user, "Admin"):
        return db_template
    elif has_permission(current_user, "Consultant"):
        if db_template.is_active:
            return db_template
        else:
            raise HTTPException(status_code=404, detail="Template not found")
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view templates"
        )

@router.put("/{template_id}", response_model=schemas.TemplateResponse)
def update_template(
    template_id: int,
    template: schemas.TemplateUpdate,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    """
    Update template by ID. Admin permission required.
    """
    if not has_permission(current_user, "Admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update templates"
        )

    db_template = db.query(entities.Template).filter(entities.Template.template_id == template_id).first()
    if db_template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    
    update_data = template.dict(exclude_unset=True)
    
    if "qa_pairs" in update_data:
        # Clear existing Q&A pairs
        for qa in db_template.qa_pairs:
            db.delete(qa)
            
        # Add new Q&A pairs
        for qa_data in update_data["qa_pairs"]:
            db_qa = entities.Template_QA(
                **qa_data,
                template_id=db_template.template_id
            )
            db.add(db_qa)
            
        del update_data["qa_pairs"]

    for key, value in update_data.items():
        setattr(db_template, key, value)
        
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    return db_template

@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_templates(
    payload: schemas.TemplateDelete,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    """
    Soft delete templates by ID. Admin permission required.
    """
    if not has_permission(current_user, "Admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete templates"
        )
    
    templates = db.query(entities.Template).filter(entities.Template.template_id.in_(payload.template_ids)).all()
    
    if not templates:
        raise HTTPException(status_code=404, detail="No templates found")
        
    for template in templates:
        template.is_active = False
        
    db.commit()
    return
