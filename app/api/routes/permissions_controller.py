from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.entities import Users
from app.core.security import has_permission, get_current_user

router = APIRouter()

@router.get("/")
def get_all_permissions(db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Get all permissions.
    """
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")
    from app.models.entities import Permission as PermissionModel
    
    permissions = db.query(PermissionModel).all()
    return permissions