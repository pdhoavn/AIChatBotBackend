from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, APIRouter, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from pathlib import Path
import os
import json
import uuid
from datetime import datetime

from app.models.database import init_db, get_db
from app.models.schemas import TrainingQuestionRequest, TrainingQuestionResponse, KnowledgeBaseDocumentResponse
from app.models import entities
from app.services.training_service import TrainingService
from app.utils.document_processor import documentProcessor
from app.core.security import get_current_user, has_permission

router = APIRouter()

def check_view_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to view training questions (Admin, Consultant, or Admission Official)"""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    try:
        user_perms_list = [p.permission_name.lower() for p in current_user.permissions] 
    except AttributeError:
        user_perms_list = [p.lower() for p in current_user.permissions]

    # Check for Admin, Consultant, or Admission Official permissions
    is_admin = "admin" in user_perms_list
    is_consultant = "consultant" in user_perms_list
    is_admission_official = any(
        "admission" in p or "admission official" in p 
        for p in user_perms_list
    )

    if not (is_admin or is_consultant or is_admission_official):
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Admission Official permission required"
        )
    
    return current_user

def get_document_or_404(document_id: int, db: Session) -> entities.KnowledgeBaseDocument:
    """Helper function to get document by ID or raise 404"""
    document = db.query(entities.KnowledgeBaseDocument).filter(
        entities.KnowledgeBaseDocument.document_id == document_id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document

def resolve_file_path(relative_path: str) -> Path:
    """
    Resolve file path relative to project root, ensuring compatibility across machines.
    Handles both absolute and relative paths stored in database.
    """
    path = Path(relative_path)
    
    # If it's already an absolute path, extract just the relative part from 'uploads/'
    if path.is_absolute():
        # Find 'uploads' in the path and take everything from there
        parts = path.parts
        try:
            uploads_index = parts.index('uploads')
            relative_path = str(Path(*parts[uploads_index:]))
            path = Path(relative_path)
        except ValueError:
            # 'uploads' not in path, use as-is
            pass
    
    # If path is not absolute, resolve it relative to current working directory
    if not path.is_absolute():
        path = Path.cwd() / path
    
    return path

def check_file_exists(file_path: str) -> Path:
    """
    Helper function to check if file exists on disk or raise 404.
    Returns the resolved absolute path.
    """
    resolved_path = resolve_file_path(file_path)
    if not resolved_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"File not found on server. Looking for: {resolved_path}"
        )
    return resolved_path

@router.post("/upload/training_question")
def api_create_training_qa(
    payload: TrainingQuestionRequest,
    db: Session = Depends(get_db),
    current_user_id: int = 1
):
    service = TrainingService()

    qa = service.create_training_qa(
        db=db,
        intent_id=payload.intent_id,
        question=payload.question,
        answer=payload.answer,
        created_by=current_user_id
    )

    return {
        "message": "Training QA created as draft",
        "qa_id": qa.question_id,
        "status": qa.status
    }
@router.post("/upload/document")
async def upload_document(
    intend_id: int = Query(...),
    file: UploadFile = File(...),
    title: str = Form(None),
    category: str = Form(None),
    current_user_id: int = Form(1),
    db: Session = Depends(get_db)
):
    print(f"\n[1] BẮT ĐẦU REQUEST Upload. Filename: {file.filename}", flush=True)
    # STEP 1: VALIDATE FILE
    try:
        print("[2] Đang gọi validate_file...", flush=True)
        is_valid, error_msg = documentProcessor.validate_file(file.filename, file.content_type)
        if not is_valid:
            print(f"[FAIL] Validate thất bại: {error_msg}", flush=True)
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        print(f"[ERROR] Lỗi tại bước Validate: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))
    
    # STEP 2: READ FILE
    try:
        print("[3] Đang đọc file vào RAM (await file.read())...", flush=True)
        file_content = await file.read()
        print(f"[4] Đọc file XONG. Kích thước: {len(file_content)} bytes", flush=True)
        if len(file_content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc file: {e}", flush=True)
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # STEP 3: EXTRACT TEXT
    try:
        print("[5] Đang gọi extract_text (Xử lý PDF/OCR)...", flush=True)
        extracted_text = documentProcessor.extract_text(
            file_content,
            file.filename,
            file.content_type
        )
        print(f"[6] Extract XONG. Kết quả text dài: {len(extracted_text) if extracted_text else 0}", flush=True)
        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Cannot extract content from the file"
            )
    except Exception as e:
        print(f"[ERROR] Lỗi tại extract_text: {e}", flush=True)
        raise HTTPException(status_code=422, detail=f"Extract error: {str(e)}")

    # STEP 4: SAVE FILE TO DISK
    try:
        print("[7] Đang lưu file xuống đĩa...", flush=True)
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = upload_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_content)
        print(f"[8] Lưu file XONG tại: {file_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] Lỗi khi lưu đĩa: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # STEP 5: SAVE DATABASE ONLY (NO QDRANT)
    try:
        service = TrainingService()
        print("[9] Đang lưu vào Database...", flush=True)
        doc = service.create_document(
            db=db,
            title=file.filename,
            file_path=str(file_path),       # <-- file text chứ không phải file gốc
            intend_id=intend_id,
            created_by=current_user_id
        )

        # save extracted text for approval stage
        temp_store_path = f"uploads/temp_text_{doc.document_id}.txt"
        with open(temp_store_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

    except Exception as e:
        print(f"[ERROR] Lỗi Database: {e}", flush=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    return {
        "message": "Document uploaded as draft. Waiting for approval.",
        "document_id": doc.document_id,
        "intend_id": doc.intend_id,
        "status": doc.status
    }


@router.get("/training_questions", response_model=List[TrainingQuestionResponse])
def get_all_training_questions(
    status: Optional[str] = Query(None, description="Filter by status: draft, approved, rejected, deleted"),
    db: Session = Depends(get_db), 
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get all training questions in the system.
    Requires Admin, Consultant, or Admission permission.
    
    - All users can see all questions regardless of status
    - Use ?status= query parameter to filter by specific status
    """
    # Build query
    query = db.query(entities.TrainingQuestionAnswer).options(joinedload(entities.TrainingQuestionAnswer.intent))
    
    # Apply status filter if provided
    if status:
        query = query.filter(entities.TrainingQuestionAnswer.status == status)
    
    training_questions = query.all()
    
    # Convert to response format
    result = []
    for tqa in training_questions:
        result.append({
            "question_id": tqa.question_id,
            "question": tqa.question,
            "answer": tqa.answer,
            "intent_id": tqa.intent_id,
            "intent_name": tqa.intent.intent_name,
            "status": tqa.status,
            "created_at": tqa.created_at,
            "approved_at": tqa.approved_at,
            "created_by": tqa.created_by,
            "approved_by": tqa.approved_by,
            "reject_reason": getattr(tqa, 'reject_reason', None)
        })
    
    return result

@router.get("/documents", response_model=List[KnowledgeBaseDocumentResponse])
def get_all_documents(
    status: Optional[str] = Query(None, description="Filter by status: draft, approved, rejected, deleted"),
    db: Session = Depends(get_db), 
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get all documents in the knowledge base.
    Requires Admin, Consultant, or Admission permission.
    
    - All users can see all documents regardless of status
    - Use ?status= query parameter to filter by specific status
    """
    # Build query
    query = db.query(entities.KnowledgeBaseDocument)
    
    # Apply status filter if provided
    if status:
        query = query.filter(entities.KnowledgeBaseDocument.status == status)
    
    documents = query.all()
    
    # Convert to response format
    result = []
    for doc in documents:
        result.append({
            "document_id": doc.document_id,
            "title": doc.title,
            "file_path": doc.file_path,
            "category": doc.category,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "created_by": doc.created_by,
            "status": doc.status,
            "reviewed_by": doc.reviewed_by,
            "reviewed_at": doc.reviewed_at,
            "reject_reason": getattr(doc, 'reject_reason', None)
        })
    
    return result

@router.get("/documents/{document_id}/download")
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Download a specific document by its ID.
    Requires Admin, Consultant, or Admission permission.
    """
    document = get_document_or_404(document_id, db)
    resolved_path = check_file_exists(document.file_path)
    
    return FileResponse(
        path=str(resolved_path),
        filename=document.title,
        media_type='application/octet-stream'
    )

@router.get("/documents/{document_id}/view")
def view_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    View/preview a specific document by its ID in browser.
    Requires Admin, Consultant, or Admission permission.
    """
    document = get_document_or_404(document_id, db)
    check_file_exists(document.file_path)
    
    # Determine media type based on file extension
    file_extension = Path(document.file_path).suffix.lower()
    media_type_mapping = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif'
    }
    
    media_type = media_type_mapping.get(file_extension, 'application/octet-stream')
    
    return FileResponse(
        path=document.file_path,
        media_type=media_type,
        headers={"Content-Disposition": "inline"}
    )

@router.get("/documents/{document_id}", response_model=KnowledgeBaseDocumentResponse)
def get_document_by_id(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Get a specific document's metadata by its ID.
    Requires Admin, Consultant, or Admission permission.
    """
    document = get_document_or_404(document_id, db)
    
    return {
        "document_id": document.document_id,
        "title": document.title,
        "file_path": document.file_path,
        "category": document.category,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "created_by": document.created_by,
        "status": document.status,
        "reviewed_by": document.reviewed_by,
        "reviewed_at": document.reviewed_at,
        "reject_reason": getattr(document, 'reject_reason', None)
    }


# ==================== REVIEW WORKFLOW ENDPOINTS ====================

def is_admin_or_leader(user: entities.Users) -> bool:
    """Helper function to check if user is Admin or Consultant Leader"""
    if not user:
        return False
    
    # Check if user is Admin (using permissions, not role)
    try:
        user_perms = [p.permission_name.lower() for p in user.permissions] 
    except AttributeError:
        user_perms = [p.lower() for p in user.permissions]
    
    is_admin = "admin" in user_perms
    
    # Check if user has Consultant permission AND is_leader flag
    has_consultant_perm = "consultant" in user_perms
    is_consultant_leader = (
        has_consultant_perm and
        user.consultant_profile is not None and 
        user.consultant_profile.is_leader
    )
    
    print(f"DEBUG is_admin_or_leader: user_id={user.user_id}, is_admin={is_admin}, is_consultant_leader={is_consultant_leader}")
    if has_consultant_perm and user.consultant_profile:
        print(f"DEBUG is_admin_or_leader: consultant_profile.is_leader={user.consultant_profile.is_leader}")
    
    return is_admin or is_consultant_leader

def check_leader_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user is Admin or Consultant Leader"""
    if not is_admin_or_leader(current_user):
        raise HTTPException(status_code=403, detail="Only Admin or Consultant Leader can review content")
    return current_user


@router.get("/documents/pending-review", response_model=List[KnowledgeBaseDocumentResponse])
def get_pending_documents(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Get all documents pending review (status=draft).
    Only Admin or ConsultantLeader can access this endpoint.
    """
    documents = db.query(entities.KnowledgeBaseDocument).filter(
        entities.KnowledgeBaseDocument.status == 'draft'
    ).all()
    
    return documents


@router.post("/documents/{document_id}/submit-review")
def submit_document_for_review(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Submit a document for review by changing status to 'draft'.
    Any consultant can submit their own documents for review.
    """
    document = get_document_or_404(document_id, db)
    
    # Check if user owns this document or is admin/leader
    if document.created_by != current_user.user_id and not is_admin_or_leader(current_user):
        raise HTTPException(status_code=403, detail="You can only submit your own documents for review")
    
    document.status = 'draft'
    db.commit()
    
    return {"message": "Document submitted for review successfully", "document_id": document_id}

@router.post("/documents/{document_id}/approve")
def api_approve_document(
    document_id: int,

    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    try:
        # Get document to retrieve its intent_id
        document = get_document_or_404(document_id, db)
        
        service = TrainingService()
        print(f"Approving document ID: {document_id}, Intent ID: {document.intend_id}")

        result = service.approve_document(
            db=db,
            document_id=document_id,
            reviewer_id=current_user.user_id,
            intent_id=document.intend_id  # Use actual intent_id from document
        )

        return {
            "message": "Document approved and indexed",
            "document_id": result.get("document_id"),
            "status": result.get("status")
        }
    except Exception as e:
        import traceback
        print(f"Error approving document {document_id}:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/documents/{document_id}/reject")
def reject_document(
    document_id: int,
    reason: str = Form(..., description="Reason for rejection"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Reject a document with a reason.
    Only Admin or ConsultantLeader can reject documents.
    """
    document = get_document_or_404(document_id, db)
    
    document.status = 'rejected'
    document.reviewed_by = current_user.user_id
    document.reviewed_at = datetime.now().date()
    document.reject_reason = reason  # Save the rejection reason
    db.commit()
    
    return {
        "message": "Document rejected",
        "document_id": document_id,
        "reason": reason,
        "reviewed_by": current_user.user_id
    }


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    service = TrainingService()
    """
    Soft delete a document by setting status to 'deleted'.
    Only Admin or ConsultantLeader can delete documents.
    """
    document = get_document_or_404(document_id, db)
    service.delete_document(db, document_id)
    document.status = 'deleted'
    db.commit()
    
    return {"message": "Document deleted successfully", "document_id": document_id}


# ==================== TRAINING Q&A REVIEW WORKFLOW ====================

def get_training_qa_or_404(question_id: int, db: Session) -> entities.TrainingQuestionAnswer:
    """Helper function to get training Q&A or raise 404"""
    qa = db.query(entities.TrainingQuestionAnswer).options(joinedload(entities.TrainingQuestionAnswer.intent)).filter(
        entities.TrainingQuestionAnswer.question_id == question_id
    ).first()
    
    if not qa:
        raise HTTPException(status_code=404, detail=f"Training Q&A with id {question_id} not found")
    
    return qa


@router.get("/training_questions/pending-review", response_model=List[TrainingQuestionResponse])
def get_pending_training_questions(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Get all training Q&A pending review (status=draft).
    Only Admin or ConsultantLeader can access this endpoint.
    """
    questions = db.query(entities.TrainingQuestionAnswer).options(joinedload(entities.TrainingQuestionAnswer.intent)).filter(
        entities.TrainingQuestionAnswer.status == 'draft'
    ).all()
    
    
    return [
        {
            "question_id": q.question_id,
            "question": q.question,
            "answer": q.answer,
            "intent_id": q.intent_id,
            "intent_name": q.intent.intent_name if q.intent else None,
            "status": q.status,
            "created_at": q.created_at,
            "approved_at": q.approved_at,
            "created_by": q.created_by,
            "approved_by": q.approved_by,
            "reject_reason": getattr(q, "reject_reason", None),
        }
        for q in questions
    ]


@router.post("/training_questions/{question_id}/submit-review")
def submit_training_qa_for_review(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission)
):
    """
    Submit a training Q&A for review by changing status to 'draft'.
    Any consultant can submit their own Q&A for review.
    """
    qa = get_training_qa_or_404(question_id, db)
    
    # Check if user owns this Q&A or is admin/leader
    if qa.created_by != current_user.user_id and not is_admin_or_leader(current_user):
        raise HTTPException(status_code=403, detail="You can only submit your own Q&A for review")
    
    qa.status = 'draft'
    db.commit()
    
    return {"message": "Training Q&A submitted for review successfully", "question_id": question_id}

@router.post("/training_questions/{question_id}/approve")
def api_approve_training_qa(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    try:
        service = TrainingService()
        print(f"Approving training question ID: {question_id}")
        result = service.approve_training_qa(
            db=db,
            qa_id=question_id,
            reviewer_id=current_user.user_id
        )

        return {
            "message": "Training QA approved",
            **result
        }
    except Exception as e:
        import traceback
        print(f"Error approving training question {question_id}:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/training_questions/{question_id}/approve")
# def approve_training_qa(
#     question_id: int,
#     db: Session = Depends(get_db),
#     current_user: entities.Users = Depends(check_leader_permission)
# ):
#     """
#     Approve a training Q&A for use in chatbot training.
#     Only Admin or ConsultantLeader can approve Q&A.
#     """
#     qa = get_training_qa_or_404(question_id, db)
    
#     qa.status = 'approved'
#     qa.approved_by = current_user.user_id
#     qa.approved_at = datetime.now().date()
#     db.commit()
    
#     return {
#         "message": "Training Q&A approved successfully",
#         "question_id": question_id,
#         "approved_by": current_user.user_id
#     }


@router.post("/training_questions/{question_id}/reject")
def reject_training_qa(
    question_id: int,
    reason: str = Form(..., description="Reason for rejection"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    """
    Reject a training Q&A with a reason.
    Only Admin or ConsultantLeader can reject Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)
    
    qa.status = 'rejected'
    qa.reject_reason = reason
    # do not reuse approved_by/approved_at for rejection
    db.commit()

    return {
        "message": "Training Q&A rejected",
        "question_id": question_id,
        "reason": reason,
        "rejected_by": current_user.user_id
    }


@router.delete("/training_questions/{question_id}")
def delete_training_qa(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission)
):
    service = TrainingService()
    """
    Soft delete a training Q&A by setting status to 'deleted'.
    Only Admin or ConsultantLeader can delete Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)
    service.delete_training_qa(db, question_id)
    qa.status = 'deleted'
    db.commit()
    
    return {"message": "Training Q&A deleted successfully", "question_id": question_id}
