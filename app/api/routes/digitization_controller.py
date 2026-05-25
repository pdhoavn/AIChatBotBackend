from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Depends,
    HTTPException,
    Form,
    Query,
    Header,
)
from fastapi.responses import JSONResponse, FileResponse
from urllib.parse import quote
import secrets
import time

# Temporary download tokens: token → {doc_id, exp}
_dl_tokens: dict = {}
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from pathlib import Path
import os
import uuid
import threading
from datetime import datetime

from app.models.database import get_db
from app.models import entities
from app.models.schemas import (
    OcrFolderCreate,
    OcrFolderUpdate,
    OcrFolderResponse,
    OcrDocumentResponse,
    OcrDocumentListResponse,
)
from app.core.security import get_current_user

router = APIRouter()


def check_digitization_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to access digitization feature"""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")
    return current_user


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify the API Key passed in the request header X-API-Key"""
    from app.core.config import settings
    if not x_api_key or x_api_key != settings.EXTERNAL_OCR_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")
    return x_api_key


# ==================== FOLDER ENDPOINTS ====================

@router.post("/folders", response_model=OcrFolderResponse)
def create_folder(
    payload: OcrFolderCreate,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Create a new OCR folder"""
    folder = entities.OcrFolder(
        folder_name=payload.folder_name,
        parent_id=payload.parent_id,
        created_by=current_user.user_id,
    )
    db.add(folder)
    db.commit()
    db.refresh(folder)
    return folder


@router.get("/folders", response_model=List[OcrFolderResponse])
def list_folders(
    parent_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """List all OCR folders, optionally filtered by parent_id"""
    query = db.query(entities.OcrFolder)
    if parent_id is not None:
        query = query.filter(entities.OcrFolder.parent_id == parent_id)
    else:
        query = query.filter(entities.OcrFolder.parent_id.is_(None))
    folders = query.all()
    return folders


@router.get("/folders/tree", response_model=List[OcrFolderResponse])
def get_folder_tree(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Get full folder tree structure"""
    def build_tree(parent_id=None):
        folders = db.query(entities.OcrFolder).filter(
            entities.OcrFolder.parent_id == parent_id
        ).all()
        result = []
        for f in folders:
            node = OcrFolderResponse.model_validate(f)
            node.children = build_tree(f.folder_id)
            result.append(node)
        return result

    return build_tree()


@router.put("/folders/{folder_id}", response_model=OcrFolderResponse)
def update_folder(
    folder_id: int,
    payload: OcrFolderUpdate,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Update folder name"""
    folder = db.query(entities.OcrFolder).filter(
        entities.OcrFolder.folder_id == folder_id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    folder.folder_name = payload.folder_name
    db.commit()
    db.refresh(folder)
    return folder


@router.delete("/folders/{folder_id}")
def delete_folder(
    folder_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Delete folder and all its documents"""
    folder = db.query(entities.OcrFolder).filter(
        entities.OcrFolder.folder_id == folder_id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    db.delete(folder)
    db.commit()
    return {"message": "Folder deleted successfully"}


# ==================== DOCUMENT ENDPOINTS ====================

@router.post("/documents/upload")
async def upload_ocr_document(
    folder_id: Optional[int] = Form(None),
    full_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Upload a document for OCR processing"""
    ext = Path(file.filename).suffix.lower()
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")

    file_content = await file.read()
    if len(file_content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    upload_dir = Path("uploads/ocr")
    upload_dir.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = upload_dir / unique_filename
    with open(file_path, "wb") as f:
        f.write(file_content)

    file_type = ext.upper().lstrip('.')

    doc = entities.OcrDocument(
        file_name=file.filename,
        file_path=str(file_path),
        file_type=file_type,
        full_name=full_name,
        folder_id=folder_id,
        status="pending",
        created_by=current_user.user_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "message": "Document uploaded successfully",
        "document_id": doc.document_id,
        "status": doc.status,
        "created_by": current_user.user_id,
        "creator_name": current_user.full_name,
        "file_name": doc.file_name,
        "file_type": doc.file_type,
    }


def _run_ocr_background(document_id: int):
    from app.models.database import SessionLocal
    from app.core.config import settings
    import pytesseract
    from PIL import Image
    import fitz

    bdb = SessionLocal()
    try:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD_PATH

        bdoc = bdb.query(entities.OcrDocument).filter_by(document_id=document_id).first()
        if not bdoc:
            return

        ext = Path(bdoc.file_path).suffix.lower()
        file_abs = str(Path(bdoc.file_path).resolve())
        output_dir = Path("uploads/ocr/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"searchable_{uuid.uuid4().hex}.pdf"
        ocr_dpi = 150
        page_accuracies: list[float] = []

        def _ocr_page(img, lang):
            try:
                data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            except pytesseract.TesseractError:
                data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
            # Build text from words
            words = []
            confs = []
            for i, word in enumerate(data['text']):
                if word.strip():
                    words.append(word.strip())
                    conf = int(data['conf'][i])
                    if conf > 0:
                        confs.append(conf)
            text = ' '.join(words)
            avg_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
            return text, avg_conf

        if ext == '.pdf':
            src_pdf = fitz.open(file_abs)
            total = src_pdf.page_count
            bdoc.total_pages = total
            bdoc.completed_pages = 0
            bdb.commit()

            output_pdf = fitz.open()

            for idx in range(total):
                page = src_pdf[idx]
                pix = page.get_pixmap(dpi=ocr_dpi, colorspace="GRAY")
                img = Image.frombytes("L", [pix.width, pix.height], pix.samples)

                page_text, page_acc = _ocr_page(img, "vie")

                try:
                    pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang="vie")
                except pytesseract.TesseractError:
                    pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang="eng")

                tmp = fitz.open("pdf", pdf_bytes)
                output_pdf.insert_pdf(tmp)
                tmp.close()

                page_accuracies.append(page_acc)
                bdoc.completed_pages = idx + 1
                bdb.commit()

            src_pdf.close()
            output_pdf.save(
                str(output_path),
                deflate=True,
                garbage=4,
                clean=True,
            )
            output_pdf.close()
        else:
            bdoc.total_pages = 1
            bdoc.completed_pages = 0
            bdb.commit()

            img = Image.open(file_abs)
            max_dim = 2000
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

            _, page_acc = _ocr_page(img, "vie")

            try:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang="vie")
            except pytesseract.TesseractError:
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img, extension='pdf', lang="eng")

            with open(str(output_path), "wb") as f:
                f.write(pdf_bytes)

            page_accuracies.append(page_acc)
            bdoc.completed_pages = 1
            bdb.commit()

        # Average accuracy across all pages
        overall_accuracy = round(sum(page_accuracies) / len(page_accuracies), 1) if page_accuracies else 0.0

        bdoc.output_pdf_path = str(output_path)
        bdoc.ocr_accuracy = overall_accuracy
        bdoc.status = "completed"
        bdb.commit()

    except Exception as e:
        bdb.rollback()
        bdoc = bdb.query(entities.OcrDocument).filter_by(document_id=document_id).first()
        if bdoc:
            bdoc.status = "failed"
            bdoc.error_message = str(e)
            bdb.commit()
    finally:
        bdb.close()


@router.post("/documents/{document_id}/ocr")
def start_ocr(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Start OCR processing for a document"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.document_id == document_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status == "processing":
        raise HTTPException(status_code=400, detail="OCR already in progress")

    doc.status = "processing"
    db.commit()

    threading.Thread(target=_run_ocr_background, args=(document_id,), daemon=True).start()

    return {
        "message": "OCR started in background",
        "document_id": document_id,
        "status": "processing",
    }


@router.get("/documents")
def list_documents(
    folder_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """List OCR documents with filtering and pagination"""
    query = db.query(entities.OcrDocument)

    if folder_id is not None:
        query = query.filter(entities.OcrDocument.folder_id == folder_id)
    if status:
        query = query.filter(entities.OcrDocument.status == status)
    if keyword:
        query = query.filter(
            entities.OcrDocument.file_name.ilike(f"%{keyword}%")
        )

    total = query.count()
    documents = query.order_by(
        entities.OcrDocument.created_at.desc()
    ).offset((page - 1) * page_size).limit(page_size).all()

    result = []
    for doc in documents:
        created_time = doc.created_at.strftime("%H:%M:%S") if doc.created_at else None
        creator_name = doc.creator.full_name if doc.creator else None
        result.append({
            "document_id": doc.document_id,
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "full_name": doc.full_name,
            "status": doc.status,
            "created_by": doc.created_by,
            "creator_name": creator_name,
            "created_at": doc.created_at,
            "created_time": created_time,
            "folder_id": doc.folder_id,
            "total_pages": doc.total_pages or 0,
            "completed_pages": doc.completed_pages or 0,
            "ocr_accuracy": doc.ocr_accuracy,
            "error_message": doc.error_message,
        })

    return {
        "items": result,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/documents/{document_id}", response_model=OcrDocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Get a specific OCR document"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.document_id == document_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc.document_id,
        "file_name": doc.file_name,
        "file_path": doc.file_path,
        "file_type": doc.file_type,
        "full_name": doc.full_name,
        "status": doc.status,
        "created_by": doc.created_by,
        "creator_name": doc.creator.full_name if doc.creator else None,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "total_pages": doc.total_pages or 0,
        "completed_pages": doc.completed_pages or 0,
        "ocr_accuracy": doc.ocr_accuracy,
        "error_message": doc.error_message,
    }


@router.get("/documents/{document_id}/progress")
def get_progress(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Poll OCR progress"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.document_id == document_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    total = doc.total_pages or 0
    completed = doc.completed_pages or 0
    progress_percent = round(completed / total * 100) if total > 0 else 0

    return {
        "document_id": doc.document_id,
        "status": doc.status,
        "total_pages": total,
        "completed_pages": completed,
        "progress_percent": progress_percent,
        "ocr_accuracy": doc.ocr_accuracy,
        "error_message": doc.error_message,
    }


@router.post("/documents/{document_id}/prepare-download")
def prepare_download(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Create a short-lived download token for native browser download"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.document_id == document_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "completed":
        raise HTTPException(status_code=400, detail="OCR not completed")
    if not doc.output_pdf_path:
        raise HTTPException(status_code=404, detail="No output PDF found")

    token = secrets.token_urlsafe(32)
    _dl_tokens[token] = {"doc_id": document_id, "exp": time.time() + 120}  # 2 phút

    # Dọn token hết hạn
    expired = [k for k, v in _dl_tokens.items() if v["exp"] < time.time()]
    for k in expired:
        del _dl_tokens[k]

    return {"token": token}


@router.get("/documents/{document_id}/download")
def download_output_pdf(
    document_id: int,
    token: Optional[str] = Query(None),
    inline: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: Optional[entities.Users] = Depends(get_current_user),
):
    """Download the searchable PDF output (via Bearer token or temp query token)"""
    # Xác thực: Bearer auth hoặc temp token
    if token:
        td = _dl_tokens.pop(token, None)
        if not td or td["exp"] < time.time() or td["doc_id"] != document_id:
            raise HTTPException(status_code=403, detail="Invalid or expired download token")
    elif not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.document_id == document_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != "completed":
        raise HTTPException(status_code=400, detail=f"OCR not completed. Status: {doc.status}")

    if not doc.output_pdf_path:
        raise HTTPException(status_code=404, detail="No output PDF found")

    resolved_path = os.path.join(os.getcwd(), doc.output_pdf_path) if not os.path.isabs(doc.output_pdf_path) else doc.output_pdf_path
    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="Output PDF file not found on disk")

    display_name = f"{doc.file_name}_searchable.pdf"
    encoded_name = quote(display_name, safe='')
    disposition = "inline" if inline else "attachment"

    return FileResponse(
        path=resolved_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"{disposition}; filename*=UTF-8''{encoded_name}"
        },
    )


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_digitization_permission),
):
    """Delete an OCR document"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.document_id == document_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete files from disk
    for p in [doc.file_path, doc.output_pdf_path]:
        if p:
            abs_p = os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p
            if os.path.exists(abs_p):
                os.remove(abs_p)

    db.delete(doc)
    db.commit()

    return {"message": "Document deleted successfully"}


# ==================== EXTERNAL OCR ENDPOINTS ====================

@router.post("/external/ocr")
async def start_external_ocr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Upload a document from an external server and start OCR immediately, returning a unique GUID filename"""
    ext = Path(file.filename).suffix.lower()
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")

    file_content = await file.read()
    if len(file_content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    upload_dir = Path("uploads/ocr")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique GUID filename and check uniqueness in DB
    while True:
        guid_filename = f"{uuid.uuid4()}{ext}"
        exists = db.query(entities.OcrDocument).filter(
            entities.OcrDocument.file_name == guid_filename
        ).first()
        if not exists:
            # Also check if it exists on disk just to be extremely thorough
            if not (upload_dir / guid_filename).exists():
                break

    file_path = upload_dir / guid_filename
    with open(file_path, "wb") as f:
        f.write(file_content)

    file_type = ext.upper().lstrip('.')

    doc = entities.OcrDocument(
        file_name=guid_filename,
        file_path=str(file_path),
        file_type=file_type,
        full_name="External Server",
        status="processing",
        created_by=None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    threading.Thread(target=_run_ocr_background, args=(doc.document_id,), daemon=True).start()

    return {
        "message": "External OCR started successfully",
        "file_name": guid_filename,
        "status": "processing",
    }


@router.get("/external/progress")
def get_external_progress(
    file_name: str = Query(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Get the OCR progress of an external document by its GUID file name"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.file_name == file_name
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    total = doc.total_pages or 0
    completed = doc.completed_pages or 0
    progress_percent = round(completed / total * 100) if total > 0 else 0

    return {
        "file_name": doc.file_name,
        "status": doc.status,
        "total_pages": total,
        "completed_pages": completed,
        "progress_percent": progress_percent,
        "ocr_accuracy": doc.ocr_accuracy,
        "error_message": doc.error_message,
    }


@router.get("/external/download")
def download_external_pdf(
    file_name: str = Query(...),
    inline: bool = Query(False),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Get the searchable PDF content for the specified external document.
    If the document is still processing, an appropriate error is returned."""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.file_name == file_name
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status in ["pending", "processing"]:
        raise HTTPException(
            status_code=400,
            detail="Document is still being processed. Please check progress and try again later."
        )
    elif doc.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed for this document. Error: {doc.error_message}"
        )
    elif doc.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"OCR is not in completed state. Status: {doc.status}"
        )

    if not doc.output_pdf_path:
        raise HTTPException(status_code=404, detail="No searchable PDF output found")

    resolved_path = os.path.join(os.getcwd(), doc.output_pdf_path) if not os.path.isabs(doc.output_pdf_path) else doc.output_pdf_path
    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="Output PDF file not found on disk")

    base_name = Path(doc.file_name).stem
    display_name = f"{base_name}_searchable.pdf"
    encoded_name = quote(display_name, safe='')
    disposition = "inline" if inline else "attachment"

    return FileResponse(
        path=resolved_path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"{disposition}; filename*=UTF-8''{encoded_name}"
        },
    )


@router.delete("/external/delete")
def delete_external_document(
    file_name: str = Query(...),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """Delete an external OCR document and its associated files from disk using the GUID file name"""
    doc = db.query(entities.OcrDocument).filter(
        entities.OcrDocument.file_name == file_name
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete files from disk
    for p in [doc.file_path, doc.output_pdf_path]:
        if p:
            abs_p = os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p
            if os.path.exists(abs_p):
                try:
                    os.remove(abs_p)
                except Exception:
                    pass

    db.delete(doc)
    db.commit()

    return {"message": "External document and associated files deleted successfully"}
