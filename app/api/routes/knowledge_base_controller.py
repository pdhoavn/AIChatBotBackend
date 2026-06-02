from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Depends,
    HTTPException,
    APIRouter,
    Form,
    Query,
)
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session, joinedload, defer
from typing import List, Optional
from pathlib import Path
import os
import json
import uuid
from datetime import datetime
from qdrant_client import models

from app.models.database import init_db, get_db, engine
from app.models.schemas import (
    KnowledgeBaseDocumentDeletedResponse,
    KnowledgeBaseDocumentMetadataUpdate,
    TrainingQuestionDeletedResponse,
    TrainingQuestionMetadataUpdate,
    TrainingQuestionRequest,
    TrainingQuestionResponse,
    KnowledgeBaseDocumentResponse,
    IntentResponse,
)
from app.models import entities
from app.services.training_service import TrainingService
from app.utils.document_processor import documentProcessor
from app.core.security import get_current_user, has_permission

router = APIRouter()

MEDIA_TYPE_MAPPING = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
}


def check_leader_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user is Admin or Consultant Leader"""
    if not is_admin_or_leader(current_user):
        raise HTTPException(
            status_code=403, detail="Only Admin or Consultant Leader can review content"
        )
    return current_user


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
        "admission" in p or "admission official" in p for p in user_perms_list
    )

    if not (is_admin or is_consultant or is_admission_official):
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Admission Official permission required",
        )

    return current_user


def get_document_or_404(
    document_id: int, db: Session
) -> entities.KnowledgeBaseDocument:
    """Helper function to get document by ID or raise 404"""
    document = (
        db.query(entities.KnowledgeBaseDocument)
        .filter(entities.KnowledgeBaseDocument.document_id == document_id)
        .first()
    )

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
            uploads_index = parts.index("uploads")
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
            detail=f"File not found on server. Looking for: {resolved_path}",
        )
    return resolved_path


def check_file_exists_public(file_path: str) -> Path:
    """
    Public-safe variant that avoids leaking absolute server paths.
    """
    resolved_path = resolve_file_path(file_path)
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")
    return resolved_path


def _resolve_metadata_audiences(db: Session, target_audiences: List[str]):
    names = target_audiences or []
    if not names:
        return [], [], []

    audiences = (
        db.query(entities.TargetAudience)
        .filter(entities.TargetAudience.name.in_(names))
        .all()
    )
    found_names = {audience.name for audience in audiences}
    missing = [name for name in names if name not in found_names]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target audiences: {', '.join(missing)}",
        )

    return (
        [audience.id for audience in audiences],
        [audience.name for audience in audiences],
        [audience.present_name for audience in audiences],
    )


def _resolve_metadata_intent(db: Session, intent_id: Optional[int]):
    if intent_id is None or intent_id == 0:
        return None

    intent = db.query(entities.Intent).filter_by(intent_id=intent_id).first()
    if not intent:
        raise HTTPException(status_code=400, detail="Invalid intent_id")
    return intent


def _set_qdrant_payload_by_filter(
    collection_name: str, field_key: str, field_value: int, payload: dict
):
    if not payload:
        return

    service = TrainingService()
    service.qdrant_client.set_payload(
        collection_name=collection_name,
        payload=payload,
        points=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field_key,
                        match=models.MatchValue(value=field_value),
                    )
                ]
            )
        ),
    )


@router.post("/upload/training_question")
def api_create_training_qa(
    payload: TrainingQuestionRequest,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    service = TrainingService()
    current_user_id = current_user.user_id
    qa = service.create_training_qa(
        db=db,
        intent_id=payload.intent_id,
        question=payload.question,
        answer=payload.answer,
        target_audiences=payload.target_audiences or [],
        created_by=current_user_id,
        is_private=payload.is_private,
    )

    return {
        "message": "Training QA created as draft",
        "qa_id": qa.question_id,
        "status": qa.status,
    }


@router.post("/upload/document")
async def upload_document(
    intend_id: int = Query(...),
    file: UploadFile = File(...),
    title: str = Form(None),
    is_private: str = Form(None),
    category: str = Form(None),
    target_audiences: List[str] = Form([]),
    current_user: entities.Users = Depends(check_leader_permission),
    db: Session = Depends(get_db),
):
    print(f"\n[1] BẮT ĐẦU REQUEST Upload. Filename: {file.filename}", flush=True)
    # STEP 1: VALIDATE FILE
    try:
        print("[2] Đang gọi validate_file...", flush=True)
        is_valid, error_msg = documentProcessor.validate_file(
            file.filename, file.content_type
        )
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
            file_content, file.filename, file.content_type
        )
        print(
            f"[6] Extract XONG. Kết quả text dài: {len(extracted_text) if extracted_text else 0}",
            flush=True,
        )
        if not extracted_text:
            # Check if PDF is a scanned image
            if (
                file.content_type == "application/pdf"
                or Path(file.filename).suffix.lower() == ".pdf"
            ):
                return JSONResponse(
                    status_code=422,
                    content={
                        "status": "SCANNED_PDF",
                        "detail": "File PDF chứa ảnh scan, không thể trích xuất nội dung. Vui lòng upload file PDF có nội dung văn bản.",
                    },
                )
            raise HTTPException(
                status_code=422, detail="Cannot extract content from the file"
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
        doc_title = (
            title.strip() if title and title.strip() else (file.filename or "Untitled")
        )
        current_user_id = current_user.user_id
        print(f"User id submid: {current_user_id}")
        # Convert is_private string to boolean
        is_private_bool = is_private.lower() in ("true", "1") if is_private else False

        doc = service.create_document(
            db=db,
            title=doc_title,
            file_path=str(file_path),
            intend_id=intend_id,
            target_audiences=target_audiences,
            created_by=current_user_id,
            is_private=is_private_bool,
            content=extracted_text,
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
        "status": doc.status,
    }


@router.post("/upload/document-ocr")
async def upload_document_ocr(
    intend_id: int = Query(...),
    file: UploadFile = File(...),
    title: str = Form(None),
    is_private: str = Form(None),
    target_audiences: List[str] = Form([]),
    current_user: entities.Users = Depends(check_leader_permission),
    db: Session = Depends(get_db),
):
    """
    Upload a scanned PDF and OCR its content in background.
    Returns immediately with document_id and task_id.
    Poll GET /knowledge/documents/{id}/task-status for progress.
    """
    ext = Path(file.filename).suffix.lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="OCR only supports PDF files")

    file_content = await file.read()
    if len(file_content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / unique_filename
    with open(file_path, "wb") as f:
        f.write(file_content)

    doc_title = (
        title.strip() if title and title.strip() else (file.filename or "Untitled")
    )
    current_user_id = current_user.user_id
    print(f"User_id: {current_user_id}")
    # Convert is_private string to boolean
    is_private_bool = is_private.lower() in ("true", "1") if is_private else False

    # Save doc as draft
    service = TrainingService()
    doc = service.create_document(
        db=db,
        title=doc_title,
        file_path=str(file_path),
        intend_id=intend_id,
        target_audiences=target_audiences,
        created_by=current_user_id,
        is_private=is_private_bool,
        content=None,
        is_ocr=True,
    )

    # Create task record
    task = entities.DocumentTask(
        document_id=doc.document_id,
        task_type="ocr",
        status="pending",
        progress=0,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Start background OCR
    import threading

    def _run_ocr():
        from app.models.database import SessionLocal

        bdb = SessionLocal()
        try:
            btask = (
                bdb.query(entities.DocumentTask).filter_by(task_id=task.task_id).first()
            )
            if not btask:
                return
            btask.status = "processing"
            bdb.commit()

            import fitz, pytesseract
            from PIL import Image
            from app.core.config import settings
            import io

            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD_PATH

            pdf_doc = fitz.open(stream=file_content, filetype="pdf")
            total = pdf_doc.page_count
            btask.total_items = total
            btask.completed_items = 0
            bdb.commit()

            all_text: list[str] = []

            def _ocr(img, lang):
                try:
                    return pytesseract.image_to_string(img, lang=lang)
                except pytesseract.TesseractError:
                    return pytesseract.image_to_string(img, lang="eng")

            for idx in range(total):
                page = pdf_doc[idx]
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = _ocr(img, "eng+vie")
                if page_text and page_text.strip():
                    all_text.append(f"\n--- Page {idx + 1} ---\n{page_text}")
                btask.completed_items = idx + 1
                btask.progress = round((idx + 1) / total * 100)
                bdb.commit()

            pdf_doc.close()
            full_text = "\n\n".join(all_text)

            # Store content
            MAX_CONTENT_DB = 50000
            if len(full_text) > MAX_CONTENT_DB:
                txt_path = str(upload_dir / f"ocr_text_{uuid.uuid4().hex}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                bdoc = (
                    bdb.query(entities.KnowledgeBaseDocument)
                    .filter_by(document_id=doc.document_id)
                    .first()
                )
                bdoc.path_txt = txt_path
            else:
                bdoc = (
                    bdb.query(entities.KnowledgeBaseDocument)
                    .filter_by(document_id=doc.document_id)
                    .first()
                )
                bdoc.content = full_text

            btask.status = "completed"
            btask.progress = 100
            bdb.commit()

        except Exception as e:
            bdb.rollback()
            btask = (
                bdb.query(entities.DocumentTask).filter_by(task_id=task.task_id).first()
            )
            if btask:
                btask.status = "failed"
                btask.error_message = str(e)
                bdb.commit()
        finally:
            bdb.close()

    threading.Thread(target=_run_ocr, daemon=True).start()

    return {
        "message": "OCR started in background",
        "document_id": doc.document_id,
        "task_id": task.task_id,
        "status": task.status,
    }


@router.get(
    "/training_questions/deleted", response_model=List[TrainingQuestionDeletedResponse]
)
def get_deleted_training_questions(db: Session = Depends(get_db)):
    service = TrainingService()
    return service.get_deleted_questions(db)


@router.get(
    "/documents/deleted", response_model=List[KnowledgeBaseDocumentDeletedResponse]
)
def get_deleted_documents(db: Session = Depends(get_db)):
    service = TrainingService()
    return service.get_deleted_documents(db)


@router.get("/training_questions", response_model=List[TrainingQuestionResponse])
def get_all_training_questions(
    status: Optional[str] = Query(
        None, description="Filter by status: draft, approved, rejected, deleted"
    ),
    is_private: Optional[bool] = Query(
        None, description="Filter by privacy: true for private, false for public"
    ),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
):
    """
    Get all training questions in the system.
    Requires Admin, Consultant, or Admission permission.

    - All users can see all questions regardless of status
    - Use ?status= query parameter to filter by specific status
    - Use ?is_private=true for private questions, ?is_private=false for public questions
    """
    # Build query
    query = db.query(entities.TrainingQuestionAnswer).options(
        joinedload(entities.TrainingQuestionAnswer.intent)
    )
    query = query.filter(entities.TrainingQuestionAnswer.status != "deleted")
    # Apply status filter if provided
    if status:
        query = query.filter(entities.TrainingQuestionAnswer.status == status)
    if is_private is True:
        query = query.filter(entities.TrainingQuestionAnswer.is_private.is_(True))
    elif is_private is False:
        query = query.filter(
            (entities.TrainingQuestionAnswer.is_private.is_(False))
            | (entities.TrainingQuestionAnswer.is_private.is_(None))
        )

    training_questions = query.all()

    # Convert to response format
    result = []
    for tqa in training_questions:
        result.append(
            {
                "question_id": tqa.question_id,
                "question": tqa.question,
                "answer": tqa.answer,
                "intent_id": tqa.intent_id,
                "intent_name": tqa.intent.intent_name if tqa.intent else None,
                "status": tqa.status,
                "created_at": tqa.created_at.date() if tqa.created_at else None,
                "approved_at": tqa.approved_at.date() if tqa.approved_at else None,
                "created_by_name": (
                    tqa.created_by_user.full_name if tqa.created_by_user else None
                ),
                "approved_by_name": (
                    tqa.approved_by_user.full_name if tqa.approved_by_user else None
                ),
                "reject_reason": getattr(tqa, "reject_reason", None),
                "target_audiences": getattr(tqa, "target_audiences", []),
                "is_private": getattr(tqa, "is_private", False),
            }
        )

    return result


@router.get("/documents", response_model=List[KnowledgeBaseDocumentResponse])
def get_all_documents(
    status: Optional[str] = Query(
        None, description="Filter by status: draft, approved, rejected, deleted"
    ),
    is_private: Optional[bool] = Query(
        None, description="Filter by privacy: true for private, false for public"
    ),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
):
    """
    Get all documents in the knowledge base.
    Requires Admin, Consultant, or Admission permission.

    - All users can see all documents regardless of status
    - Use ?status= query parameter to filter by specific status
    - Use ?is_private=true for private documents, ?is_private=false for public documents
    """
    # Select only list-view fields. Use an autocommit connection for this read-only
    # listing because this local Postgres connection can time out on transaction
    # commit/rollback after larger read queries.
    sql = """
        SELECT
            d.document_id,
            d.title,
            d.file_path,
            d.category,
            d.created_at,
            d.updated_at,
            d.created_by,
            d.status,
            d.is_private,
            d.is_ocr,
            d.reviewed_by,
            d.reviewed_at,
            d.reject_reason,
            d.target_audiences,
            d.intend_id AS intent_id,
            i.intent_name,
            author.full_name AS created_by_name,
            reviewer.full_name AS reviewed_by_name
        FROM "KnowledgeBaseDocument" d
        LEFT JOIN "Intent" i ON i.intent_id = d.intend_id
        LEFT JOIN "Users" author ON author.user_id = d.created_by
        LEFT JOIN "Users" reviewer ON reviewer.user_id = d.reviewed_by
        WHERE d.status != :deleted_status
    """
    params = {"deleted_status": "deleted"}
    if status:
        sql += " AND d.status = :status"
        params["status"] = status
    if is_private is True:
        sql += " AND d.is_private IS TRUE"
    elif is_private is False:
        sql += " AND COALESCE(d.is_private, false) IS FALSE"

    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        documents = conn.execute(text(sql), params).mappings().all()
    finally:
        conn.invalidate()
        conn.close()

    # Convert to response format
    result = []
    for doc in documents:
        result.append(
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "file_path": doc.file_path,
                "category": doc.category,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at,
                "created_by": doc.created_by,
                "status": doc.status,
                "is_private": bool(doc.is_private),
                "is_ocr": doc.is_ocr,
                "reviewed_by": doc.reviewed_by,
                "created_by_name": doc.created_by_name,
                "reviewed_at": doc.reviewed_at,
                "reviewed_by_name": doc.reviewed_by_name,
                "reject_reason": doc.reject_reason,
                "target_audiences": doc.target_audiences or [],
                "intent_id": doc.intent_id,
                "intent_name": doc.intent_name,
            }
        )

    return result


@router.get("/intentbyid", response_model=List[IntentResponse])
def get_categories(
    target_audience: str = Query(
        None, description="Filter by target audience (e.g., 'Viên chức/Ngưới lao động')"
    ),
    db: Session = Depends(get_db),
):
    """
    Get distinct intents linked to:
    - knowledge base documents
    - training questions
    Optionally filter by target_audience to only return intents relevant to the selected audience.
    """
    # Intent IDs from documents
    doc_intent_ids = db.query(
        entities.KnowledgeBaseDocument.intend_id.label("intent_id")
    ).filter(entities.KnowledgeBaseDocument.intend_id.isnot(None))
    if target_audience:
        doc_intent_ids = doc_intent_ids.filter(
            entities.KnowledgeBaseDocument.target_audiences.any(target_audience)
        )

    # Intent IDs from training questions
    qa_intent_ids = db.query(
        entities.TrainingQuestionAnswer.intent_id.label("intent_id")
    ).filter(entities.TrainingQuestionAnswer.intent_id.isnot(None))
    if target_audience:
        qa_intent_ids = qa_intent_ids.filter(
            entities.TrainingQuestionAnswer.target_audiences.any(target_audience)
        )

    # Union intent IDs from both sources, then fetch Intent records
    all_intent_ids = doc_intent_ids.union(qa_intent_ids).subquery()
    intents = (
        db.query(entities.Intent)
        .join(all_intent_ids, entities.Intent.intent_id == all_intent_ids.c.intent_id)
        .all()
    )
    return intents


@router.get("/documents/{document_id}/download")
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
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
        media_type="application/octet-stream",
    )


@router.get("/documents/{document_id}/view")
def view_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
):
    """
    View/preview a specific document by its ID in browser.
    Requires Admin, Consultant, or Admission permission.
    """
    document = get_document_or_404(document_id, db)
    check_file_exists(document.file_path)

    # Determine media type based on file extension
    file_extension = Path(document.file_path).suffix.lower()
    media_type = MEDIA_TYPE_MAPPING.get(file_extension, "application/octet-stream")

    return FileResponse(
        path=document.file_path,
        media_type=media_type,
        headers={"Content-Disposition": "inline"},
    )


@router.get("/documents/{document_id}/public-view")
def public_view_document(document_id: int, db: Session = Depends(get_db)):
    """
    Public, read-only document view endpoint.
    Security constraints:
    - No authentication required
    - Only approved documents are visible
    - Inline response only (no metadata disclosure)
    """
    document = get_document_or_404(document_id, db)

    if document.status != "approved":
        raise HTTPException(status_code=404, detail="Document not available")

    resolved_path = check_file_exists_public(document.file_path)
    file_extension = resolved_path.suffix.lower()
    media_type = MEDIA_TYPE_MAPPING.get(file_extension, "application/octet-stream")

    return FileResponse(
        path=str(resolved_path),
        media_type=media_type,
        headers={"Content-Disposition": "inline"},
    )


@router.get("/documents/{document_id}", response_model=KnowledgeBaseDocumentResponse)
def get_document_by_id(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
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
        "created_at": document.created_at.date() if document.created_at else None,
        "updated_at": document.updated_at.date() if document.updated_at else None,
        "created_by": document.created_by,
        "created_by_name": document.author.full_name if document.author else None,
        "reviewed_by_name": document.reviewer.full_name if document.reviewer else None,
        "status": document.status,
        "reviewed_by": document.reviewed_by,
        "reviewed_at": document.reviewed_at.date() if document.reviewed_at else None,
        "reject_reason": getattr(document, "reject_reason", None),
        "target_audiences": getattr(document, "target_audiences", []),
        "is_private": getattr(document, "is_private", False),
        "intent_id": document.intent.intent_id if document.intent else None,
        "intent_name": document.intent.intent_name if document.intent else None,
    }


@router.patch(
    "/documents/{document_id}/metadata", response_model=KnowledgeBaseDocumentResponse
)
def update_document_metadata(
    document_id: int,
    payload: KnowledgeBaseDocumentMetadataUpdate,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Update document metadata only. This endpoint never changes extracted content,
    chunks, or embeddings. For approved documents, it updates Qdrant payload
    fields used by RAG filters without re-embedding the document content.
    """
    document = get_document_or_404(document_id, db)
    update_data = payload.dict(exclude_unset=True)

    if "title" in update_data:
        title = (payload.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        document.title = title

    if "category" in update_data:
        document.category = payload.category

    intent = None
    if "intent_id" in update_data:
        intent = _resolve_metadata_intent(db, payload.intent_id)
        document.intend_id = intent.intent_id if intent else None
    else:
        intent = document.intent

    qdrant_payload = {}
    if "intent_id" in update_data:
        qdrant_payload["intent_id"] = intent.intent_id if intent else 0
        qdrant_payload["intent_name"] = intent.intent_name if intent else None

    if "is_private" in update_data:
        document.is_private = bool(payload.is_private)
        qdrant_payload["is_private"] = bool(payload.is_private)

    if "target_audiences" in update_data:
        audience_ids, _, audience_present_names = _resolve_metadata_audiences(
            db, payload.target_audiences or []
        )
        document.target_audiences = payload.target_audiences or []
        qdrant_payload["audience_ids"] = audience_ids
        qdrant_payload["audience_names"] = audience_present_names

    document.updated_at = datetime.now()

    try:
        if document.status == "approved" and qdrant_payload:
            _set_qdrant_payload_by_filter(
                "knowledge_base_documents",
                "document_id",
                document_id,
                qdrant_payload,
            )
        db.commit()
        db.refresh(document)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Could not update document metadata: {exc}",
        ) from exc

    return {
        "document_id": document.document_id,
        "title": document.title,
        "file_path": document.file_path,
        "category": document.category,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "created_by": document.created_by,
        "created_by_name": document.author.full_name if document.author else None,
        "status": document.status,
        "reviewed_by": document.reviewed_by,
        "reviewed_at": document.reviewed_at,
        "reviewed_by_name": document.reviewer.full_name if document.reviewer else None,
        "reject_reason": getattr(document, "reject_reason", None),
        "target_audiences": getattr(document, "target_audiences", []),
        "is_private": getattr(document, "is_private", False),
        "is_ocr": getattr(document, "is_ocr", False),
        "intent_id": document.intent.intent_id if document.intent else None,
        "intent_name": document.intent.intent_name if document.intent else None,
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
        has_consultant_perm
        and user.consultant_profile is not None
        and user.consultant_profile.is_leader
    )

    print(
        f"DEBUG is_admin_or_leader: user_id={user.user_id}, is_admin={is_admin}, is_consultant_leader={is_consultant_leader}"
    )
    if has_consultant_perm and user.consultant_profile:
        print(
            f"DEBUG is_admin_or_leader: consultant_profile.is_leader={user.consultant_profile.is_leader}"
        )

    return is_admin or is_consultant_leader


@router.get(
    "/documents/pending-review", response_model=List[KnowledgeBaseDocumentResponse]
)
def get_pending_documents(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Get all documents pending review (status=draft).
    Only Admin or ConsultantLeader can access this endpoint.
    """
    documents = (
        db.query(entities.KnowledgeBaseDocument)
        .options(
            joinedload(entities.KnowledgeBaseDocument.intent),
            defer(entities.KnowledgeBaseDocument.content),
        )
        .filter(entities.KnowledgeBaseDocument.status == "draft")
        .all()
    )
    return [
        {
            "document_id": doc.document_id,
            "title": doc.title,
            "file_path": doc.file_path,
            "category": doc.category,
            "created_at": doc.created_at.date() if doc.created_at else None,
            "updated_at": doc.updated_at.date() if doc.updated_at else None,
            "created_by": doc.created_by,
            "status": doc.status,
            "reviewed_by": doc.reviewed_by,
            "reviewed_at": doc.reviewed_at.date() if doc.reviewed_at else None,
            "created_by_name": doc.author.full_name if doc.author else None,
            "reviewed_by_name": doc.reviewer.full_name if doc.reviewer else None,
            "reject_reason": getattr(doc, "reject_reason", None),
            "target_audiences": getattr(doc, "target_audiences", []),
            "is_private": getattr(doc, "is_private", False),
            "intent_id": doc.intent.intent_id if doc.intent else None,
            "intent_name": doc.intent.intent_name if doc.intent else None,
        }
        for doc in documents
    ]


@router.post("/documents/{document_id}/submit-review")
def submit_document_for_review(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
):
    """
    Submit a document for review by changing status to 'draft'.
    Any consultant can submit their own documents for review.
    """
    document = get_document_or_404(document_id, db)

    # Check if user owns this document or is admin/leader
    if document.created_by != current_user.user_id and not is_admin_or_leader(
        current_user
    ):
        raise HTTPException(
            status_code=403, detail="You can only submit your own documents for review"
        )

    document.status = "draft"
    db.commit()

    return {
        "message": "Document submitted for review successfully",
        "document_id": document_id,
    }


@router.post("/documents/{document_id}/approve")
def api_approve_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Approve a document and index its chunks into Qdrant in background.
    Returns immediately. Poll GET /knowledge/documents/{id}/task-status for progress.
    """
    document = get_document_or_404(document_id, db)
    if document.status != "draft":
        raise HTTPException(
            status_code=400, detail="Only draft documents can be approved"
        )

    # Create task record
    task = entities.DocumentTask(
        document_id=document_id,
        task_type="approve",
        status="pending",
        progress=0,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    # Start background processing
    import threading

    def _run_approve():
        import time as _time
        from app.models.database import SessionLocal

        bdb = SessionLocal()
        reviewer_id = current_user.user_id
        try:
            btask = (
                bdb.query(entities.DocumentTask).filter_by(task_id=task.task_id).first()
            )
            if not btask:
                return
            btask.status = "processing"
            bdb.commit()

            bdoc = (
                bdb.query(entities.KnowledgeBaseDocument)
                .filter_by(document_id=document_id)
                .first()
            )
            if not bdoc:
                btask.status = "failed"
                btask.error_message = "Document not found"
                bdb.commit()
                return

            # Audience validation
            audience_names_input = bdoc.target_audiences or []
            audiences = (
                bdb.query(entities.TargetAudience)
                .filter(entities.TargetAudience.name.in_(audience_names_input))
                .all()
            )
            if not audiences:
                btask.status = "failed"
                btask.error_message = "No valid audiences found"
                bdb.commit()
                return

            audience_ids = [a.id for a in audiences]
            filtered_audience_names = [a.present_name for a in audiences]

            intent = (
                bdb.query(entities.Intent).filter_by(intent_id=bdoc.intend_id).first()
            )

            # Get content
            # content = getattr(bdoc, "content", None)
            # if not content:
            #     txt_path = getattr(bdoc, "path_txt", None)
            #     if txt_path:
            #         resolved = (
            #             os.path.join(os.getcwd(), txt_path)
            #             if not os.path.isabs(txt_path)
            #             else txt_path
            #         )
            #         if os.path.exists(resolved):
            #             with open(resolved, "r", encoding="utf-8") as f:
            #                 content = f.read()

            # if not content:
            #     btask.status = "failed"
            #     btask.error_message = "No content to index"
            #     bdb.commit()
            #     return

            service = TrainingService()
            ext = os.path.splitext(bdoc.file_path)[1].lower()

            try:
                chunks, use_header_split = service._extract_and_chunk(bdoc, ext)
                # In ra để xem chunk thực tế có to như cấu hình không
                print(f"DEBUG CHUNKING: Tổng số {len(chunks)} chunks được tạo.")
                if chunks:
                    print(
                        f"DEBUG CHUNKING: Độ dài chunk đầu tiên: {len(chunks[0])} ký tự."
                    )
                doc_context = "\n".join(c for c in chunks[:5] if "| --- |" not in c)[
                    :1500
                ]
                chunks = service._enrich_table_chunks(chunks, doc_context)
                print(f"DEBUG ENRICH: Enrich xong, tổng {len(chunks)} chunks.")
            except Exception as e:
                btask.status = "failed"
                btask.error_message = f"Extraction failed: {str(e)}"
                bdb.commit()
                return

            if not chunks:
                btask.status = "failed"
                btask.error_message = "No content to index"
                bdb.commit()
                return

            total = len(chunks)
            btask.total_items = total
            bdb.commit()

            # Split
            # from langchain_text_splitters import RecursiveCharacterTextSplitter

            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=1000, chunk_overlap=200
            # )
            # chunks = text_splitter.split_text(content)
            # total = len(chunks)
            # btask.total_items = total
            # bdb.commit()

            # # Embed + index
            # service = TrainingService()
            from qdrant_client.models import PointStruct

            for i, chunk in enumerate(chunks):
                embedding = service.embeddings.embed_query(chunk)
                point_id = str(uuid.uuid4())
                service.qdrant_client.upsert(
                    collection_name="knowledge_base_documents",
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "document_id": document_id,
                                "chunk_index": i,
                                "chunk_text": chunk,
                                "audience_ids": audience_ids,
                                "audience_names": filtered_audience_names,
                                "intent_id": bdoc.intend_id,
                                "intent_name": intent.intent_name if intent else None,
                                "type": "document",
                                "is_private": bdoc.is_private or False,
                            },
                        )
                    ],
                )
                btask.completed_items = i + 1
                btask.progress = round((i + 1) / total * 100)
                bdb.commit()

            # Finalize
            bdoc.status = "approved"
            bdoc.reviewed_by = reviewer_id
            bdoc.reviewed_at = datetime.now()
            btask.status = "completed"
            btask.progress = 100
            bdb.commit()

        except Exception as e:
            bdb.rollback()
            btask = (
                bdb.query(entities.DocumentTask).filter_by(task_id=task.task_id).first()
            )
            if btask:
                btask.status = "failed"
                btask.error_message = str(e)
                bdb.commit()
        finally:
            bdb.close()

    threading.Thread(target=_run_approve, daemon=True).start()

    return {
        "message": "Approval started in background",
        "document_id": document_id,
        "task_id": task.task_id,
        "status": task.status,
    }


@router.post("/documents/{document_id}/reject")
def reject_document(
    document_id: int,
    reason: str = Form(..., description="Reason for rejection"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Reject a document with a reason.
    Only Admin or ConsultantLeader can reject documents.
    """
    document = get_document_or_404(document_id, db)

    document.status = "rejected"
    document.reviewed_by = current_user.user_id
    document.reviewed_at = datetime.now().date()
    document.reject_reason = reason  # Save the rejection reason
    db.commit()

    return {
        "message": "Document rejected",
        "document_id": document_id,
        "reason": reason,
        "reviewed_by": current_user.user_id,
    }


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    service = TrainingService()
    """
    Soft delete a document by setting status to 'deleted'.
    Only Admin or ConsultantLeader can delete documents.
    """
    document = get_document_or_404(document_id, db)
    service.delete_document(db, document_id, current_user)
    document.status = "deleted"
    db.commit()

    return {"message": "Document deleted successfully", "document_id": document_id}


# ==================== TRAINING Q&A REVIEW WORKFLOW ====================


def get_training_qa_or_404(
    question_id: int, db: Session
) -> entities.TrainingQuestionAnswer:
    """Helper function to get training Q&A or raise 404"""
    qa = (
        db.query(entities.TrainingQuestionAnswer)
        .options(joinedload(entities.TrainingQuestionAnswer.intent))
        .filter(entities.TrainingQuestionAnswer.question_id == question_id)
        .first()
    )

    if not qa:
        raise HTTPException(
            status_code=404, detail=f"Training Q&A with id {question_id} not found"
        )

    return qa


@router.get(
    "/training_questions/pending-review", response_model=List[TrainingQuestionResponse]
)
def get_pending_training_questions(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Get all training Q&A pending review (status=draft).
    Only Admin or ConsultantLeader can access this endpoint.
    """
    questions = (
        db.query(entities.TrainingQuestionAnswer)
        .options(joinedload(entities.TrainingQuestionAnswer.intent))
        .filter(entities.TrainingQuestionAnswer.status == "draft")
        .all()
    )

    return [
        {
            "question_id": q.question_id,
            "question": q.question,
            "answer": q.answer,
            "intent_id": q.intent_id,
            "intent_name": q.intent.intent_name if q.intent else None,
            "status": q.status,
            "created_at": q.created_at.date() if q.created_at else None,
            "approved_at": q.approved_at.date() if q.approved_at else None,
            "created_by": q.created_by,
            "approved_by": q.approved_by,
            "created_by_name": (
                q.created_by_user.full_name if q.created_by_user else None
            ),
            "approved_by_name": (
                q.approved_by_user.full_name if q.approved_by_user else None
            ),
            "reject_reason": getattr(q, "reject_reason", None),
            "target_audiences": getattr(q, "target_audiences", []),
            "is_private": getattr(q, "is_private", False),
        }
        for q in questions
    ]


@router.post("/training_questions/{question_id}/submit-review")
def submit_training_qa_for_review(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_view_permission),
):
    """
    Submit a training Q&A for review by changing status to 'draft'.
    Any consultant can submit their own Q&A for review.
    """
    qa = get_training_qa_or_404(question_id, db)

    # Check if user owns this Q&A or is admin/leader
    if qa.created_by != current_user.user_id and not is_admin_or_leader(current_user):
        raise HTTPException(
            status_code=403, detail="You can only submit your own Q&A for review"
        )

    qa.status = "draft"
    db.commit()

    return {
        "message": "Training Q&A submitted for review successfully",
        "question_id": question_id,
    }


@router.patch(
    "/training_questions/{question_id}/metadata",
    response_model=TrainingQuestionResponse,
)
def update_training_question_metadata(
    question_id: int,
    payload: TrainingQuestionMetadataUpdate,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Update training Q&A metadata only. This endpoint never changes question or
    answer text, so existing embeddings remain valid. For approved questions,
    it updates Qdrant payload fields used by RAG filters without re-embedding.
    """
    qa = get_training_qa_or_404(question_id, db)
    update_data = payload.dict(exclude_unset=True)

    intent = None
    if "intent_id" in update_data:
        intent = _resolve_metadata_intent(db, payload.intent_id)
        qa.intent_id = intent.intent_id if intent else None
    else:
        intent = qa.intent

    qdrant_payload = {}
    if "intent_id" in update_data:
        qdrant_payload["intent_id"] = intent.intent_id if intent else 0
        qdrant_payload["intent_name"] = intent.intent_name if intent else None

    if "is_private" in update_data:
        qa.is_private = bool(payload.is_private)
        qdrant_payload["is_private"] = bool(payload.is_private)

    if "target_audiences" in update_data:
        audience_ids, _, audience_present_names = _resolve_metadata_audiences(
            db, payload.target_audiences or []
        )
        qa.target_audiences = payload.target_audiences or []
        qdrant_payload["audience_ids"] = audience_ids
        qdrant_payload["audience_names"] = audience_present_names

    try:
        if qa.status == "approved" and qdrant_payload:
            _set_qdrant_payload_by_filter(
                "training_qa",
                "question_id",
                question_id,
                qdrant_payload,
            )
        db.commit()
        db.refresh(qa)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Could not update question metadata: {exc}",
        ) from exc

    return {
        "question_id": qa.question_id,
        "question": qa.question,
        "answer": qa.answer,
        "intent_id": qa.intent_id,
        "intent_name": qa.intent.intent_name if qa.intent else None,
        "status": qa.status,
        "created_at": qa.created_at,
        "approved_at": qa.approved_at,
        "created_by": qa.created_by,
        "approved_by": qa.approved_by,
        "created_by_name": qa.created_by_user.full_name if qa.created_by_user else None,
        "approved_by_name": (
            qa.approved_by_user.full_name if qa.approved_by_user else None
        ),
        "reject_reason": getattr(qa, "reject_reason", None),
        "target_audiences": getattr(qa, "target_audiences", []),
        "is_private": getattr(qa, "is_private", False),
    }


@router.post("/training_questions/{question_id}/approve")
def api_approve_training_qa(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    try:
        service = TrainingService()
        print(f"Approving training question ID: {question_id}")
        result = service.approve_training_qa(
            db=db, qa_id=question_id, reviewer_id=current_user.user_id
        )

        return {"message": "Training QA approved", **result}
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
    current_user: entities.Users = Depends(check_leader_permission),
):
    """
    Reject a training Q&A with a reason.
    Only Admin or ConsultantLeader can reject Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)

    qa.status = "rejected"
    qa.reject_reason = reason
    # do not reuse approved_by/approved_at for rejection
    db.commit()

    return {
        "message": "Training Q&A rejected",
        "question_id": question_id,
        "reason": reason,
        "rejected_by": current_user.user_id,
    }


@router.delete("/training_questions/{question_id}")
def delete_training_qa(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_leader_permission),
):
    service = TrainingService()
    """
    Soft delete a training Q&A by setting status to 'deleted'.
    Only Admin or ConsultantLeader can delete Q&A.
    """
    qa = get_training_qa_or_404(question_id, db)
    service.delete_training_qa(db, question_id, current_user)
    qa.status = "deleted"
    db.commit()

    return {"message": "Training Q&A deleted successfully", "question_id": question_id}
