from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.security import get_current_user, verify_user_access
from app.models.database import get_db
from app.models.schemas import UserProfileResponse
from app.models.entities import Users, Role

router = APIRouter()

@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: int,
    current_user: Users = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user profile information. Users can only access their own profile.
    """
    try:
        print(f"Starting profile fetch for user_id: {user_id}")
        print(f"Current user: {current_user.user_id if current_user else 'None'}")
        
        # Check if user has permission to access this profile
        verify_user_access(current_user.user_id, user_id)
        print("Access verification passed")

        # Get user with role information (using outer join to support null role_id)
        user = db.query(Users).outerjoin(Role).filter(Users.user_id == user_id).first()
        print(f"User query result: {user}")
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        print(f"User found: {user.full_name}, email: {user.email}")
        
        # Build basic profile response
        profile_data = {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "phone_number": user.phone_number,
            "permission": [],
            "role_name": None,
            "student_profile": None,
            "consultant_profile": None,
            "content_manager_profile": None,
            "admission_official_profile": None
        }
        
        print("Basic profile data created")
        
        # Safely get permissions
        try:
            if user.permissions:
                profile_data["permission"] = [permission.permission_name for permission in user.permissions if permission.permission_name]
                print(f"Permissions added: {profile_data['permission']}")
        except Exception as e:
            print(f"Error getting permissions: {e}")
            profile_data["permission"] = []
            
        # Safely get role
        try:
            if user.role:
                profile_data["role_name"] = user.role.role_name
                print(f"Role added: {profile_data['role_name']}")
        except Exception as e:
            print(f"Error getting role: {e}")
            profile_data["role_name"] = None

        # Safely get profiles - simplified for debugging
        try:
            if hasattr(user, 'customer_profile') and user.customer_profile:
                customer_profile = user.customer_profile
                student_profile_data = {}

                # Safely get interest data
                if customer_profile.interest:
                    student_profile_data["interest"] = {
                        "interest_id": customer_profile.interest.interest_id,
                        "desired_major": customer_profile.interest.desired_major,
                        "region": customer_profile.interest.region
                    }
                else:
                    student_profile_data["interest"] = None
                
                profile_data["student_profile"] = student_profile_data
                print("Student profile exists with interest data")
        except Exception as e:
            print(f"Error with student profile: {e}")
            
        try:
            if hasattr(user, 'consultant_profile') and user.consultant_profile:
                profile_data["consultant_profile"] = {
                    "status": user.consultant_profile.status,
                    "is_leader": user.consultant_profile.is_leader
                }
                print("Consultant profile exists")
        except Exception as e:
            print(f"Error with consultant profile: {e}")
            
        try:
            if hasattr(user, 'content_manager_profile') and user.content_manager_profile:
                profile_data["content_manager_profile"] = {
                    "is_leader": user.content_manager_profile.is_leader
                }
                profile_data["content_manager_is_leader"] = user.content_manager_profile.is_leader
                print(f"Content manager profile exists, is_leader: {user.content_manager_profile.is_leader}")
        except Exception as e:
            print(f"Error with content manager profile: {e}")
            
        try:
            if hasattr(user, 'consultant_profile') and user.consultant_profile:
                profile_data["consultant_profile"] = {
                    "status": user.consultant_profile.status,
                    "is_leader": user.consultant_profile.is_leader
                }
                profile_data["consultant_is_leader"] = user.consultant_profile.is_leader
                print(f"Consultant profile exists, is_leader: {user.consultant_profile.is_leader}")
        except Exception as e:
            print(f"Error with consultant profile: {e}")
            
        try:
            if hasattr(user, 'admission_official_profile') and user.admission_official_profile:
                profile_data["admission_official_profile"] = {
                    "rating": user.admission_official_profile.rating,
                    "current_sessions": user.admission_official_profile.current_sessions,
                    "max_sessions": user.admission_official_profile.max_sessions,
                    "status": user.admission_official_profile.status
                }
                print("Admission official profile exists")
        except Exception as e:
            print(f"Error with admission official profile: {e}")

        print(f"Final profile data: {profile_data}")
        return profile_data
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 403, 404)
        raise
    except Exception as e:
        # Log the actual error for debugging
        print(f"CRITICAL ERROR in get_user_profile: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error while fetching profile: {str(e)}"
        )