from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import schemas, database, entities
from app.core.security import get_current_user
from typing import Optional

from app.services.training_service import TrainingService

router = APIRouter()

@router.post("/submit")
async def submit_riasec(
    riasec_result: schemas.RiasecResultCreate,
    db: Session = Depends(database.get_db),
    current_user: Optional[entities.Users] = Depends(get_current_user),
   
):
    """
    Logic đơn giản:
    - Luôn tạo summary bằng LLM.
    - Nếu có user đăng nhập → lưu vào DB.
    - Nếu không → chỉ trả summary (không lưu DB).
    """
    service = TrainingService()
    # 1) Gọi LLM để tạo summary RIASEC
    summary_text = await service.response_from_riasec_result(riasec_result)

    # Nếu user chưa login → trả summary, không lưu gì
    if current_user is None:
        return {
            "summary": summary_text,
            "scores": riasec_result
        }

    # 2) User có login → lưu vào DB
    try:
        # Check if CustomerProfile exists, create if not
        customer_profile = db.query(entities.CustomerProfile).filter(
            entities.CustomerProfile.customer_id == current_user.user_id
        ).first()
        
        if not customer_profile:
            # Create CustomerProfile for this user
            customer_profile = entities.CustomerProfile(
                customer_id=current_user.user_id,
                interest_id=None
            )
            db.add(customer_profile)
            db.flush()  # Ensure the profile is created before adding RIASEC result
        
        new_result = entities.RiasecResult(
            score_realistic=riasec_result.score_realistic,
            score_investigative=riasec_result.score_investigative,
            score_artistic=riasec_result.score_artistic,
            score_social=riasec_result.score_social,
            score_enterprising=riasec_result.score_enterprising,
            score_conventional=riasec_result.score_conventional,
            result=summary_text,
            customer_id=current_user.user_id
        )

        db.add(new_result)
        db.commit()
        db.refresh(new_result)

        return new_result

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Lỗi khi lưu kết quả: {str(e)}")
    

@router.get(
    "/users/{user_id}/riasec/results",
    response_model=list[schemas.RiasecResultBase]
)
def get_riasec_results(
    user_id: int, 
    db: Session = Depends(database.get_db)
):
    results = (
        db.query(entities.RiasecResult)
        .filter(entities.RiasecResult.customer_id == user_id)
        .order_by(entities.RiasecResult.result_id.desc())
        .all()
    )

    if not results:
        return []  # Trả về mảng rỗng thay vì message

    return results