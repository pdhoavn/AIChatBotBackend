from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    verify_user_access,
)
from app.models.database import get_db
from app.models.schemas import (
    Token,
    UserCreate,
    UserResponse,
    LoginRequest,
)
from app.models.entities import Users, UserPermission

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(*, db: Session = Depends(get_db), user_in: UserCreate) -> Any:
    """
    Register a new user.
    """
    user = db.query(Users).filter(Users.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    
    # Create user with basic information; role_id is deferred
    user = Users(
        email=user_in.email,
        full_name=user_in.full_name,
        phone_number=user_in.phone_number,
        password=get_password_hash(user_in.password),
        status=True,
    )
    db.add(user)
    # Flush so the new user's PK (user_id) is populated before creating UserPermission rows
    # (Without flush/commit, user.user_id will be None for an autoincrement PK.)
    db.flush()

    # Import models that we'll need in both branches
    from app.models.entities import Role as RoleModel
    
    # Add permissions if provided (validate permission ids first)
    if user_in.permissions:
        # Import models here to avoid circular import at module load
        from app.models.entities import Permission as PermissionModel
        from app.models.entities import ConsultantProfile as ConsultantProfileModel
        from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
        from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

        # Validate permissions and get their names
        perms = db.query(PermissionModel).filter(PermissionModel.permission_id.in_(user_in.permissions)).all()
        if len(perms) != len(set(user_in.permissions)):
            db.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more permission IDs are invalid.")

        permission_names = { (p.permission_name or "").lower() for p in perms }

        # Assign permissions to user
        for perm in perms:
            db.add(UserPermission(user_id=user.user_id, permission_id=perm.permission_id))

        # Determine and set the user's role based on permissions
        if any("admission" in name for name in permission_names):
            admission_role = db.query(RoleModel).filter(RoleModel.role_name.ilike("%admission%")).first()
            if admission_role:
                user.role_id = admission_role.role_id
            else:
                user.role_id = None  # Or handle missing "Admission" role error
        else:
            # If permissions are given but none are admission, role is explicitly null
            user.role_id = None
        
        # Create related profiles based on granted permissions
        # Consultant profile
        if any(name for name in permission_names if "consultant" in name):
            consultant_profile = ConsultantProfileModel(
                consultant_id=user.user_id,
                # ConsultantProfile.status already defaults to True in the model, but set explicitly
                status=True,
                is_leader=bool(getattr(user_in, "consultant_is_leader", False))
            )
            db.add(consultant_profile)

        # Content manager profile
        if any(name for name in permission_names if "content" in name or "content_manager" in name or "content manager" in name):
            content_manager_profile = ContentManagerProfileModel(
                content_manager_id=user.user_id,
                is_leader=bool(getattr(user_in, "content_manager_is_leader", False))
            )
            db.add(content_manager_profile)

        # Admission official profile
        if any(name for name in permission_names if "admission" in name or "official" in name or "admission_official" in name):
            admission_profile = AdmissionOfficialProfileModel(
                admission_official_id=user.user_id,
                rating=0,
                current_sessions=0,
                max_sessions=10,
                status="available"
            )
            db.add(admission_profile)
    else:
        # No permissions provided => regular customer user
        # Find or create a "Customer" role
        customer_role = db.query(RoleModel).filter(RoleModel.role_name.ilike("customer")).first()
        if not customer_role:
            # Create Customer role if it doesn't exist
            customer_role = RoleModel(role_name="Customer")
            db.add(customer_role)
            db.flush()  # Get the role_id
        
        user.role_id = customer_role.role_id
        
        # Create CustomerProfile for this user
        # Optionally create an Interest record if interest data was provided during registration
        from app.models.entities import CustomerProfile as CustomerProfileModel
        from app.models.entities import Interest as InterestModel

        interest_obj = None
        if getattr(user_in, "interest_desired_major", None) or getattr(user_in, "interest_region", None):
            interest_obj = InterestModel(
                desired_major=getattr(user_in, "interest_desired_major", None),
                region=getattr(user_in, "interest_region", None),
            )
            db.add(interest_obj)
            # flush so interest_id is populated
            db.flush()

        customer_profile = CustomerProfileModel(
            customer_id=user.user_id,
            interest_id=interest_obj.interest_id if interest_obj else None
        )
        db.add(customer_profile)

    db.commit()
    db.refresh(user)
    
    # Prepare response with permissions
    response = {
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "phone_number": user.phone_number,
        "status": user.status,
        "role_id": user.role_id,
        "permissions": [p.permission_id for p in user.permissions] if user.permissions else []
    }
    return response


@router.post("/login", response_model=Token)
def login(
    db: Session = Depends(get_db),
    form_data: LoginRequest = None
) -> Any:
    """
    Login to get an access token for future requests.
    """
    if not form_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login credentials required",
        )
    
    email = form_data.email
    password = form_data.password

    user = db.query(Users).filter(Users.email == email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.status:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Your account has been deactivated. Please contact the administrator."
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            {"sub": user.email, "user_id": user.user_id}
        ),
            "token_type": "bearer",
        }
