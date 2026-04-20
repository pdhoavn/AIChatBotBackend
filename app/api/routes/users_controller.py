from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload
from app.models.database import get_db
from app.models.schemas import (
    PermissionChangeRequest,
    PermissionRevokeRequest,
    BanUserRequest,
    UserUpdate,
)
from app.models.entities import Users, UserPermission, Permission
from app.core.security import has_permission, get_current_user, is_admin_or_admission_official
from sqlalchemy import not_, or_

router = APIRouter()


@router.get("/permissions")
def get_all_permissions(db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Get all available permissions in the system.
    Requires admin permission.
    """
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required",
        )

    permissions = db.query(Permission).all()
    return [{"permission_id": p.permission_id, "permission_name": p.permission_name} for p in permissions]


@router.get("/roles")
def get_all_roles(db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Get all available roles in the system.
    Requires authentication.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    from app.models.entities import Role
    roles = db.query(Role).all()
    return [{"role_id": r.role_id, "role_name": r.role_name} for r in roles]


@router.get("/students")
def get_students(db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Get all customer users (Students and Parents - users without permissions).
    Requires admin or admission official permission.
    """
    if not current_user or not is_admin_or_admission_official(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or admission official permission required",
        )

    # Get users with no permissions (customers/students/parents)
    # Also include users with Student or Parent roles explicitly
    from app.models.entities import Role
    
    customers = db.query(Users).options(
        selectinload(Users.permissions),
        selectinload(Users.role)
    ).outerjoin(Users.role).filter(
        or_(
            not_(Users.permissions.any()),  # Users with no permissions
            Role.role_name.in_(['Student', 'Parent'])  # Or users with Student/Parent role
        )
    ).all()
    
    # Format the response to match staffs endpoint structure
    result = []
    for user in customers:
        user_data = {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "phone_number": user.phone_number,
            "status": user.status,
            "role_id": user.role_id,
            "role_name": user.role.role_name if user.role else None,  # Include role name
            "password": user.password,  # Include for compatibility with existing frontend
            "permissions": [{"permission_name": p.permission_name, "permission_id": p.permission_id} for p in user.permissions] if user.permissions else [],
        }
        result.append(user_data)
    
    return result


@router.get("/staffs")
def get_staffs(db: Session = Depends(get_db), current_user: Users = Depends(get_current_user)):
    """
    Get all users who have at least one permission (staff), except own user.
    Requires admin permission.
    """
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required",
        )

    # Query users with permissions loaded and profile relationships
    staffs = db.query(Users).options(
        selectinload(Users.permissions),
        selectinload(Users.consultant_profile),
        selectinload(Users.content_manager_profile), 
        selectinload(Users.admission_official_profile),
        selectinload(Users.role)
    ).filter(
        Users.permissions.any(), 
        Users.user_id != current_user.user_id
    ).all()
    
    # Format the response to include permission names and profile indicators
    result = []
    for user in staffs:
        user_data = {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "phone_number": user.phone_number,
            "status": user.status,
            "role_id": user.role_id,
            "role_name": user.role.role_name if user.role else None,  # Include role name
            "password": user.password,  # Include for compatibility with existing frontend
            "permissions": [{"permission_name": p.permission_name, "permission_id": p.permission_id} for p in user.permissions],
            "consultant_is_leader": user.consultant_profile.is_leader if user.consultant_profile else False,
            "content_manager_is_leader": user.content_manager_profile.is_leader if user.content_manager_profile else False,
            "consultant_profile": bool(user.consultant_profile),
            "content_manager_profile": bool(user.content_manager_profile),
            "admission_official_profile": bool(user.admission_official_profile),
        }
        result.append(user_data)
    
    return result


@router.post("/permissions/grant")
def grant_permission(
    payload: PermissionChangeRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Grant one or more permissions to a user (admin only). Returns summary of changes."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    # find target user
    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Normalize requested ids
    requested_ids = list(dict.fromkeys(payload.permission_ids or []))
    if not requested_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No permission ids provided")

    # validate permission ids exist
    from app.models.entities import Permission as PermissionModel
    from app.models.entities import ConsultantProfile as ConsultantProfileModel
    from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
    from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

    perms = db.query(PermissionModel).filter(PermissionModel.permission_id.in_(requested_ids)).all()
    found_ids = {p.permission_id for p in perms}
    missing = [pid for pid in requested_ids if pid not in found_ids]
    if missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"missing_permission_ids": missing})

    added = []
    skipped = []

    # current permission ids
    current_ids = {p.permission_id for p in (target.permissions or [])}

    # add new permissions
    add_perms = [p for p in perms if p.permission_id not in current_ids]
    for perm in add_perms:
        db.add(UserPermission(user_id=target.user_id, permission_id=perm.permission_id))
        added.append(perm.permission_id)

    # collect skipped ids (already present)
    skipped = [pid for pid in requested_ids if pid in current_ids]

    # create related profiles if necessary (once per type)
    added_names = {(p.permission_name or "").lower() for p in add_perms}
    if any("consultant" in n for n in added_names):
        cp = db.query(ConsultantProfileModel).filter(ConsultantProfileModel.consultant_id == target.user_id).first()
        if not cp:
            cp = ConsultantProfileModel(consultant_id=target.user_id, status=True, is_leader=bool(payload.consultant_is_leader))
            db.add(cp)
    if any("content" in n for n in added_names):
        cmp = db.query(ContentManagerProfileModel).filter(ContentManagerProfileModel.content_manager_id == target.user_id).first()
        if not cmp:
            cmp = ContentManagerProfileModel(content_manager_id=target.user_id, is_leader=bool(payload.content_manager_is_leader))
            db.add(cmp)
    if any(("admission" in n) or ("official" in n) for n in added_names):
        ap = db.query(AdmissionOfficialProfileModel).filter(AdmissionOfficialProfileModel.admission_official_id == target.user_id).first()
        if not ap:
            ap = AdmissionOfficialProfileModel(admission_official_id=target.user_id, rating=0, current_sessions=0, max_sessions=10, status="available")
            db.add(ap)

    # If any permissions were added, set user status to True
    if added:
        target.status = True

    db.commit()
    return {"added": added, "skipped": skipped}


@router.delete("/permissions/revoke")
def revoke_permission(
    payload: PermissionRevokeRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Revoke one or more permissions from a user (admin only). Admins cannot revoke permissions from other admins."""
    try:
        if not current_user or not has_permission(current_user, "admin"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

        # find target user
        target = db.query(Users).filter(Users.user_id == payload.user_id).first()
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

        # Determine if target has admin permission
        from app.models.entities import Permission as PermissionModel
        target_has_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in target.permissions or [])
        if target_has_admin and target.user_id != current_user.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify permissions of another admin")

        requested_ids = list(dict.fromkeys(payload.permission_ids or []))
        if not requested_ids:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No permission ids provided")

        # Validate that the permission ids exist
        perms = db.query(PermissionModel).filter(PermissionModel.permission_id.in_(requested_ids)).all()
        found_ids = {p.permission_id for p in perms}
        missing = [pid for pid in requested_ids if pid not in found_ids]
        if missing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"missing_permission_ids": missing})

        removed = []
        skipped = []

        # find existing UserPermission rows for requested ids
        ups = db.query(UserPermission).filter(
            UserPermission.user_id == target.user_id,
            UserPermission.permission_id.in_(requested_ids)
        ).all()
        existing_ids = {u.permission_id for u in ups}
        skipped = [pid for pid in requested_ids if pid not in existing_ids]

        # import profile models
        from app.models.entities import ConsultantProfile as ConsultantProfileModel
        from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
        from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

        # delete found links
        for u in ups:
            db.delete(u)
            removed.append(u.permission_id)

        db.flush()

        # After removals, recompute remaining permissions to decide profile cleanup
        remaining_perms = db.query(PermissionModel).join(UserPermission).filter(UserPermission.user_id == target.user_id).all()
        remaining_names = {(p.permission_name or "").lower() for p in remaining_perms}

        # Clean profiles if no remaining related permissions
        if not any("consultant" in name for name in remaining_names):
            cp = db.query(ConsultantProfileModel).filter(ConsultantProfileModel.consultant_id == target.user_id).first()
            if cp:
                db.delete(cp)
        if not any("content" in name for name in remaining_names):
            cmp = db.query(ContentManagerProfileModel).filter(ContentManagerProfileModel.content_manager_id == target.user_id).first()
            if cmp:
                db.delete(cmp)
        if not any(("admission" in name) or ("official" in name) for name in remaining_names):
            ap = db.query(AdmissionOfficialProfileModel).filter(AdmissionOfficialProfileModel.admission_official_id == target.user_id).first()
            if ap:
                # Check if there are any live chat queue entries referencing this officer
                from app.models.entities import LiveChatQueue
                active_queues = db.query(LiveChatQueue).filter(
                    LiveChatQueue.customer_id == target.user_id
                ).count()
                
                if active_queues > 0:
                    db.rollback()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cannot revoke admission officer permission: This user has {active_queues} active or pending live chat queue entries. Please resolve these queue entries first."
                    )
                
                db.delete(ap)

        # If user has no remaining permissions, set status to False (ban)
        remaining = db.query(UserPermission).filter(UserPermission.user_id == target.user_id).all()
        if not remaining:
            target.status = False

        db.commit()
        return {"removed": removed, "skipped": skipped}
    
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        print(f"Error in revoke_permission: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to revoke permissions: {str(e)}"
        )


@router.put("/permissions/update")
def update_permissions(
    payload: PermissionChangeRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Update/replace all permissions for a user (admin only).
    Admins cannot modify the permissions of other admins.
    """
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Prevent admin from modifying another admin's permissions
    from app.models.entities import Permission as PermissionModel, Role as RoleModel
    target_is_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in target.permissions or [])
    if target_is_admin and target.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify permissions of another admin")

    # Clear existing permissions
    db.query(UserPermission).filter(UserPermission.user_id == target.user_id).delete()

    # Validate and add new permissions
    requested_ids = list(dict.fromkeys(payload.permission_ids or []))
    if requested_ids:
        perms = db.query(PermissionModel).filter(PermissionModel.permission_id.in_(requested_ids)).all()
        found_ids = {p.permission_id for p in perms}
        missing = [pid for pid in requested_ids if pid not in found_ids]
        if missing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"missing_permission_ids": missing})

        for perm in perms:
            db.add(UserPermission(user_id=target.user_id, permission_id=perm.permission_id))

    # Check for "Admission" permission by name and update role
    has_admission_perm = any("admission" in p.permission_name.lower() for p in perms if p.permission_name)
    
    if has_admission_perm:
        # Find a role that contains "Admission" in its name, case-insensitive
        admission_role = db.query(RoleModel).filter(RoleModel.role_name.ilike("%admission%")).first()
        if admission_role:
            target.role = admission_role
        else:
            # This case might happen if the "Admission" role is missing in the DB
            target.role = None
    else:
        target.role = None

    db.commit()
    db.refresh(target)

    # Return the updated user permissions
    updated_permission_ids = [p.permission_id for p in target.permissions]
    return {"user_id": target.user_id, "permission_ids": updated_permission_ids}


@router.post("/ban")
def ban_user(
    payload: BanUserRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Ban a user (admin only). Cannot ban another admin."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Prevent banning another admin
    target_is_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in (target.permissions or []))
    if target_is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot ban another admin")

    # Check if user is an admission officer with active queue entries
    # from app.models.entities import AdmissionOfficialProfile, LiveChatQueue
    # admission_profile = db.query(AdmissionOfficialProfile).filter(
    #     AdmissionOfficialProfile.admission_official_id == target.user_id
    # ).first()
    
    # if admission_profile:
    #     # Check for any live chat queue entries
    #     active_queues = db.query(LiveChatQueue).filter(
    #         LiveChatQueue.id == target.user_id
    #     ).count()
        
    #     if active_queues > 0:
    #         raise HTTPException(
    #             status_code=status.HTTP_400_BAD_REQUEST,
    #             detail=f"Cannot ban this admission officer: They have {active_queues} active or pending live chat queue entries. Please resolve these queue entries first."
    #         )

    target.status = False
    db.commit()
    return {"message": "User has been banned"}


@router.post("/unban")
def unban_user(
    payload: BanUserRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Unban a user (admin only). Cannot unban another admin."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Prevent unbanning another admin
    target_is_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in (target.permissions or []))
    if target_is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify status of another admin")

    target.status = True
    db.commit()
    return {"message": "User has been unbanned"}


@router.get("/{user_id}")
def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Get a single user by ID.
    Requires admin or admission official permission.
    """
    if not current_user or not is_admin_or_admission_official(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or admission official permission required"
        )

    # Find user with relationships loaded
    user = db.query(Users).options(
        selectinload(Users.permissions),
        selectinload(Users.role)
    ).filter(Users.user_id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )

    # Return user data in same format as students/staffs endpoints
    return {
        "user_id": user.user_id,
        "full_name": user.full_name,
        "email": user.email,
        "phone_number": user.phone_number,
        "status": user.status,
        "role_id": user.role_id,
        "role_name": user.role.role_name if user.role else None,
        "password": user.password,  # Include for compatibility
        "permissions": [
            {
                "permission_name": p.permission_name, 
                "permission_id": p.permission_id
            } for p in user.permissions
        ] if user.permissions else [],
    }


@router.put("/{user_id}")
def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Update user basic information (admin only).
    Can update: full_name, email, phone_number, password, status.
    Cannot modify another admin's information unless updating own profile.
    """
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )

    # Find target user
    target = db.query(Users).options(
        selectinload(Users.permissions),
        selectinload(Users.role)
    ).filter(Users.user_id == user_id).first()
    
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent modifying another admin (unless it's yourself)
    target_is_admin = any(
        p.permission_name and "admin" in p.permission_name.lower() 
        for p in (target.permissions or [])
    )
    if target_is_admin and target.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify another admin's information"
        )

    # Check if email is being changed and if it's already taken
    if user_update.email and user_update.email != target.email:
        existing_user = db.query(Users).filter(
            Users.email == user_update.email,
            Users.user_id != user_id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    # Update fields that are provided
    update_data = user_update.dict(exclude_unset=True)
    
    # Check if status is being changed to False (ban) for admission officer with active queues
    if "status" in update_data and update_data["status"] == False:
        from app.models.entities import AdmissionOfficialProfile, LiveChatQueue
        admission_profile = db.query(AdmissionOfficialProfile).filter(
            AdmissionOfficialProfile.admission_official_id == target.user_id
        ).first()
        
        if admission_profile:
            active_queues = db.query(LiveChatQueue).filter(
                LiveChatQueue.admission_official_id == target.user_id
            ).count()
            
            if active_queues > 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot deactivate this admission officer: They have {active_queues} active or pending live chat queue entries. Please resolve these queue entries first."
                )
    
    # Handle password hashing if password is being updated
    if "password" in update_data and update_data["password"]:
        from app.core.security import get_password_hash
        update_data["password"] = get_password_hash(update_data["password"])
    
    # Apply updates
    for field, value in update_data.items():
        setattr(target, field, value)

    db.commit()
    db.refresh(target)

    # Return updated user info
    return {
        "user_id": target.user_id,
        "full_name": target.full_name,
        "email": target.email,
        "phone_number": target.phone_number,
        "status": target.status,
        "role_id": target.role_id,
        "role_name": target.role.role_name if target.role else None,
        "message": "User updated successfully"
    }
