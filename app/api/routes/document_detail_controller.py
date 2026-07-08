from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from pathlib import Path

from app.models.database import get_db
from app.models import entities
from app.models.schemas import (
    DocumentDetailResponse,
    DocumentContentResponse,
    DocumentChunkItemResponse,
    DocumentTaskResponse,
)
from app.infrastructure.qdrant_manager import get_qdrant_client
from app.core.security import get_current_user

router = APIRouter()


def _check_view_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to view documents (Admin, Consultant, or Admission Official)."""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    try:
        user_perms_list = [p.permission_name.lower() for p in current_user.permissions]
    except AttributeError:
        user_perms_list = [p.lower() for p in current_user.permissions]

    is_admin = "admin" in user_perms_list
    is_consultant = "consultant" in user_perms_list
    is_admission_official = any("admission" in p or "admission official" in p for p in user_perms_list)

    if not (is_admin or is_consultant or is_admission_official):
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Admission Official permission required"
        )
    return current_user


def _get_document_or_404(document_id: int, db: Session) -> entities.KnowledgeBaseDocument:
    document = (
        db.query(entities.KnowledgeBaseDocument)
        .options(joinedload(entities.KnowledgeBaseDocument.intent))
        .filter(entities.KnowledgeBaseDocument.document_id == document_id)
        .first()
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/documents/{document_id}/detail", response_model=DocumentDetailResponse)
def get_document_detail(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(_check_view_permission),
):
    """
    Get document overview: metadata, processing tasks, and counts (content, chunks, qdrant).
    """
    doc = _get_document_or_404(document_id, db)

    # Count Qdrant points
    qdrant_count = 0
    try:
        qdrant = get_qdrant_client()
        qdrant_count = qdrant.count_points(
            collection_name="knowledge_base_documents",
            document_id=document_id,
        )
    except Exception:
        pass

    content = getattr(doc, "content", None) or ""

    # Fallback: count chars from txt file if content not in DB
    content_char_count = len(content)
    if not content_char_count:
        txt_path = getattr(doc, "path_txt", None)
        if txt_path:
            try:
                resolved = Path(txt_path)
                if not resolved.is_absolute():
                    resolved = Path.cwd() / resolved
                if resolved.exists():
                    content_char_count = resolved.stat().st_size
            except Exception:
                pass

    return {
        "document_id": doc.document_id,
        "title": doc.title,
        "file_path": doc.file_path,
        "category": doc.category,
        "status": doc.status,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "created_by": doc.created_by,
        "reviewed_by": doc.reviewed_by,
        "created_by_name": doc.author.full_name if doc.author else None,          
        "reviewed_by_name": doc.reviewer.full_name if doc.reviewer else None,
        "reviewed_at": doc.reviewed_at,
        "reject_reason": getattr(doc, "reject_reason", None),
        "target_audiences": getattr(doc, "target_audiences", []),
        "target_units": getattr(doc, "target_units", []),
        "intent_id": doc.intent.intent_id if doc.intent else None,
        "intent_name": doc.intent.intent_name if doc.intent else None,
        "content_char_count": content_char_count,
        "qdrant_points_count": qdrant_count,
        "is_private": getattr(doc, "is_private", False),
        "is_ocr": getattr(doc, "is_ocr", False),
        "path_txt": getattr(doc, "path_txt", None),
    }


@router.get("/documents/{document_id}/content", response_model=DocumentContentResponse)
def get_document_content(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(_check_view_permission),
):
    """
    Get full extracted text content of a document.
    Falls back to reading from path_txt file if content column is empty.
    """
    doc = _get_document_or_404(document_id, db)
    content = getattr(doc, "content", None) or ""

    # Fallback: read from txt file if content stored externally
    if not content:
        txt_path = getattr(doc, "path_txt", None)
        if txt_path:
            try:
                resolved = Path(txt_path)
                if not resolved.is_absolute():
                    resolved = Path.cwd() / resolved
                if resolved.exists():
                    with open(resolved, "r", encoding="utf-8") as f:
                        content = f.read()
            except Exception:
                pass

    return {
        "document_id": doc.document_id,
        "content": content,
        "char_count": len(content),
    }


@router.get("/documents/{document_id}/chunks", response_model=List[DocumentChunkItemResponse])
def get_document_chunks(
    document_id: int,
    source: Optional[str] = Query("db", description="Source: 'db' for DB chunks, 'qdrant' for Qdrant points"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(_check_view_permission),
):
    """
    Get chunks of a document from DB or Qdrant.
    """
    doc = _get_document_or_404(document_id, db)

    if source == "qdrant":
        qdrant = get_qdrant_client()
        try:
            points = qdrant.scroll_document_chunks(
                collection_name="knowledge_base_documents",
                document_id=document_id,
                limit=1000,
            )
            if points:
                points.sort(key=lambda x: x.get("chunk_index") or 0)
            return [
                {
                    "chunk_id": None,
                    "point_id": str(p["point_id"]),
                    "chunk_index": p.get("chunk_index"),
                    "chunk_text": p.get("chunk_text") or "",
                    "char_count": len(p.get("chunk_text") or ""),
                }
                for p in points
            ]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Qdrant error: {str(exc)}")

    # Default: DB chunks
    chunks = (
        db.query(entities.DocumentChunk)
        .filter(entities.DocumentChunk.document_id == document_id)
        .order_by(entities.DocumentChunk.chunk_id.asc())
        .all()
    )

    return [
        {
            "chunk_id": c.chunk_id,
            "point_id": None,
            "chunk_index": idx,
            "chunk_text": c.chunk_text or "",
            "char_count": len(c.chunk_text or ""),
        }
        for idx, c in enumerate(chunks)
    ]


@router.get(
    "/documents/{document_id}/task-status",
    response_model=Optional[DocumentTaskResponse],
)
def get_task_status(
    document_id: int,
    task_type: Optional[str] = Query(None, description="Filter by type: 'ocr' or 'approve'"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(_check_view_permission),
):
    """
    Get the latest background task status for a document.
    Returns None if no task exists.
    """
    query = (
        db.query(entities.DocumentTask)
        .filter(entities.DocumentTask.document_id == document_id)
        .order_by(entities.DocumentTask.created_at.desc())
    )
    if task_type:
        query = query.filter(entities.DocumentTask.task_type == task_type)

    task = query.first()
    return task
