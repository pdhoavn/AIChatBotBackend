from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import OAuth2PasswordBearer
import os
from dotenv import load_dotenv
from app.models.schemas import TokenData
from app.models.database import get_db
from app.models.entities import Users
from sqlalchemy.orm import Session, joinedload

# Load environment variables
load_dotenv()

# JWT Configuration from .env
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt_sha256", "bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(email=email, user_id=user_id)
        return token_data
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_user_access(requesting_user_id: int, target_user_id: int):
    """
    Verify if the requesting user has access to view the target user's profile
    """
    if requesting_user_id != target_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this profile"
        )

def is_admin(user: Users) -> bool:
    """
    Check if user is admin based on permission (not role).
    Admin has full permissions.
    """
    if not user or not user.permissions:
        return False
    
    # Check if user has "admin" permission
    permission_names = {p.permission_name.lower() for p in user.permissions if p.permission_name}
    return "admin" in permission_names

def has_permission(user: Users, permission_name: str) -> bool:
    """
    Check if user has a specific permission by name.
    Admins bypass and always have permission.
    
    Args:
        user: Users object with permissions relationship loaded
        permission_name: Name of permission to check (e.g., "content_manager", "consultant", "admin")
    
    Returns:
        True if user is admin or has the permission, False otherwise
    """
    if not user:
        return False
    
    # Admin has all permissions
    if is_admin(user):
        return True
    
    # Check if user has the specific permission
    if not user.permissions:
        return False
    
    permission_names = {p.permission_name.lower() for p in user.permissions if p.permission_name}
    return permission_name.lower() in permission_names

def is_admin_or_admission_official(user: Users) -> bool:
    """Check if user is an admin or an admission official."""
    if not user:
        return False
    return has_permission(user, "Admin") or has_permission(user, "Admission Official")

def verify_content_manager(user: Users) -> bool:
    """
    Verify if user is a content manager or admin.
    Checks permissions instead of role_id.
    Admin role has full permissions.
    """
    if not user:
        return False
    
    # Admin has full access
    if is_admin(user):
        return True
    
    # Check for content manager permission
    return has_permission(user, "Content Manager")

def verify_content_manager_leader(user: Users) -> bool:
    """
    Verify if user is a content manager leader or admin.
    Checks if user has content_manager permission AND is_leader flag is true.
    Admin role bypasses this check.
    """
    if not user:
        return False
    
    # Admin has full access (bypass leader check)
    if is_admin(user):
        return True
    
    # Check if user has content_manager permission
    if not has_permission(user, "Content Manager"):
        return False
    
    # Check if is_leader flag is set
    return (user.content_manager_profile and 
            user.content_manager_profile.is_leader)

def verify_consultant(user: Users) -> bool:
    """
    Verify if user is a consultant or admin.
    Checks permissions instead of role_id.
    """
    if not user:
        return False
    
    # Admin has full access
    if is_admin(user):
        return True
    
    # Check for consultant permission
    return has_permission(user, "consultant")

def verify_consultant_leader(user: Users) -> bool:
    """
    Verify if user is a consultant leader or admin.
    Checks if user has consultant permission AND is_leader flag is true.
    """
    if not user:
        return False
    
    # Admin has full access (bypass leader check)
    if is_admin(user):
        return True
    
    # Check if user has consultant permission
    if not has_permission(user, "consultant"):
        return False
    
    # Check if is_leader flag is set
    return (user.consultant_profile and 
            user.consultant_profile.is_leader)

async def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[Users]:
    """
    Get current user from token in Authorization header.
    Returns None if no token is provided or token is invalid.
    """    
    auth_header = request.headers.get("Authorization")
    print(f"DEBUG: Auth header received: {auth_header}")
    
    if not auth_header or "Bearer" not in auth_header:
        print("DEBUG: No Authorization header or Bearer not found")
        return None
        
    try:
        token = auth_header.split(" ")[1]
        print(f"DEBUG: Token extracted: {token[:20]}...")
        
        token_data = verify_token(token)
        if token_data.email is None:
            print("DEBUG: Token verification failed - no email")
            return None
            
        print(f"DEBUG: Token verified for email: {token_data.email}")
            
        # Load user with permissions, role, and consultant_profile relationships eagerly loaded
        from sqlalchemy.orm import selectinload
        user = db.query(Users).options(
            selectinload(Users.permissions),
            selectinload(Users.role),
            selectinload(Users.consultant_profile)
        ).filter(Users.email == token_data.email).first()
        
        print(f"DEBUG: Found user: {user.email if user else 'None'}")
        
        if user:
            print(f"DEBUG: Loaded {len(user.permissions)} permissions for user {user.user_id}")
            for perm in user.permissions:
                print(f"DEBUG: Permission: {perm.permission_name}")
        
        if user is None or not user.status:
            print("DEBUG: User not found or inactive")
            return None
            
        return user
    except (JWTError, Exception) as e:
        print(f"DEBUG: Exception in get_current_user: {e}")
        return None
