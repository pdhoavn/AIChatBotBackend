from app.core.security import get_current_user
from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Request,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
from dotenv import load_dotenv
import json
import time
import uuid
from app.models.database import SessionLocal
from app.services import training_service
from app.services.training_service import TrainingService


class ChatStreamRequest(BaseModel):
    """Request body cho endpoint SSE /stream."""

    message: str
    session_id: Optional[int] = None
    user_id: Optional[int] = None
    audience_id: Optional[int] = None
    intent_id: Optional[int] = None


router = APIRouter()
# Tạo 1 instance dùng chung
service = TrainingService()
load_dotenv()


def _chat_log(message: str, trace_id: str = ""):
    if trace_id:
        print(f"[CHAT][{trace_id}] {message}")
        return
    print(f"[CHAT] {message}")


# thêm 3 tầng check chat
@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    # session_id = 1
    # user_id = 1
    top_k = os.getenv("TOP_K", 5)
    confidence_threshold = float(os.getenv("CONFIDENCE_SCORE", 0.35))
    service = TrainingService()
    await websocket.accept()

    # 1️⃣ Nhận thông tin user và session trước
    data = await websocket.receive_json()

    user_id = data.get("user_id")
    session_id = data.get("session_id")

    if not session_id:
        session_id = service.create_chat_session(user_id, "chatbot")
        await websocket.send_json(
            {"event": "session_created", "session_id": session_id}
        )

    await websocket.send_json({"event": "go", "sources": [], "confidence": 1.0})

    try:
        while True:
            # Nhận tin nhắn từ client
            raw_data = await websocket.receive_json()
            message = raw_data.get("message", "").strip()
            audience_id = raw_data.get("audience_id")
            intent_id_from_client = raw_data.get("intent_id")
            print(f"audience_id : {audience_id}, intent_id: {intent_id_from_client}")
            if not message:
                continue
            trace_id = uuid.uuid4().hex[:8]
            request_start = time.perf_counter()
            _chat_log(
                f"incoming_message session_id={session_id} user_id={user_id} message_len={len(message)}",
                trace_id,
            )

            # enrich_query — tạo truy vấn "đầy đủ" dựa vào hội thoại cũ
            enriched_query = await service.enrich_query(session_id, message)
            print(f"👉 enriched_query: {enriched_query}")
            _chat_log(f"enriched_query={enriched_query}", trace_id)

            # Nếu enrich_query rỗng, nghĩa là user nói lan man → không cần RAG
            if not enriched_query:
                _chat_log("skip_rag: empty_enriched_query", trace_id)
                await websocket.send_json(
                    {
                        "event": "chunk",
                        "content": "Mình chưa rõ ý bạn lắm, bạn có thể nói rõ hơn được không?",
                    }
                )
                await websocket.send_json(
                    {"event": "done", "sources": [], "confidence": 0.0}
                )
                continue

            # Tìm context liên quan
            # doc_results = TrainingService.search_documents(message, top_k=5)

            # Hybrid search (cả training QA và document)
            try:
                hybrid_start = time.perf_counter()
                result = await service.hybrid_search(
                    audience_ids=audience_id,
                    query=enriched_query,
                    intent_id=intent_id_from_client,
                    trace_id=trace_id,
                )
                hybrid_elapsed_ms = int((time.perf_counter() - hybrid_start) * 1000)
                _chat_log(
                    f"hybrid_search_done elapsed_ms={hybrid_elapsed_ms}", trace_id
                )

            except Exception as e:
                print(f"Hybrid search error: {e}")
                _chat_log(
                    f"hybrid_search_error type={type(e).__name__} message={e}", trace_id
                )
                await websocket.send_json(
                    {
                        "event": "chunk",
                        "content": "Hệ thống đang bận hoặc kho dữ liệu tạm thời không phản hồi. Bạn thử lại sau ít phút nhé.",
                    }
                )
                await websocket.send_json(
                    {"event": "done", "sources": [], "confidence": 0.0}
                )
                continue
            tier_source = result.get("response_source")
            confidence = result.get("confidence", 0.0)
            print(f"👉 tier_source: {tier_source}")
            print(f"👉 confidence: {confidence}")
            _chat_log(
                f"tier_source={tier_source} confidence={confidence:.6f} "
                f"sources={result.get('sources', [])}",
                trace_id,
            )
            # === TIER 1: training_qa - score > 0.8 ===
            if tier_source == "training_qa" and confidence > confidence_threshold:
                print("floor 1")
                top = result["top_match"]
                q_text = top.payload.get("question_text")
                a_text = top.payload.get("answer_text")
                intent_id = top.payload.get("intent_id")
                relevance_ok = await service.llm_relevance_check(
                    enriched_query, q_text, a_text
                )
                _chat_log(f"training_qa_relevance={relevance_ok}", trace_id)

                if relevance_ok:
                    print("floor 1: training QA valid")
                    async for chunk in service.stream_response_from_qa(
                        enriched_query,
                        a_text,
                        session_id,
                        user_id,
                        intent_id,
                        message,
                    ):
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "event": "chunk",
                                    "content": getattr(chunk, "content", str(chunk)),
                                }
                            )
                        )
                    await websocket.send_json(
                        {"event": "done", "sources": [], "confidence": confidence}
                    )
                    total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
                    _chat_log(
                        f"done tier=training_qa confidence={confidence:.6f} elapsed_ms={total_elapsed_ms}",
                        trace_id,
                    )
                    continue
                else:
                    print("QA not relevant → fallback xuống document")
                    # Chạy document search lại
                    doc_results = await service.search_documents(
                        enriched_query,
                        audience_ids=audience_id,
                        intent_id=intent_id_from_client,
                        top_k=top_k,
                        trace_id=trace_id,
                        stage="document_recheck_search",
                        query_embedding=result.get("query_embedding"),
                    )
                    result = service.build_document_search_result(doc_results)
                    confidence = result.get("confidence", 0.0)
                    tier_source = "document"
                    _chat_log(
                        f"qa_fallback_result confidence={confidence:.6f} sources={result.get('sources', [])}",
                        trace_id,
                    )

            context_chunks = result["response"]
            intent_id = result["intent_id"]
            context = "\n\n".join(
                [r.payload.get("chunk_text", "") for r in context_chunks]
            )
            _chat_log(
                f"context_precheck chunks={len(context_chunks)} chars={len(context)} intent_id={intent_id}",
                trace_id,
            )

            tier_check_start = time.perf_counter()
            tier_source = await service.llm_document_recommendation_check(
                enriched_query, context
            )
            tier_check_elapsed_ms = int((time.perf_counter() - tier_check_start) * 1000)
            _chat_log(
                f"llm_document_recommendation_check result={tier_source} elapsed_ms={tier_check_elapsed_ms}",
                trace_id,
            )
            # Context and confidence are already populated from previous search
            if tier_source == "document":
                print("Context:" + context)
                print("Confidence of document:")
                print(confidence)
            print("SOURCE NAME: " + tier_source)
            # === TIER 2: document-only (no QA match) ===
            if tier_source == "document" and confidence >= confidence_threshold:
                print("🔍 floor 3: using document context")
                answer_text = ""
                async for chunk in service.stream_response_from_context(
                    enriched_query,
                    context,
                    session_id,
                    user_id,
                    intent_id,
                    message,
                ):
                    piece = getattr(chunk, "content", str(chunk))
                    answer_text += piece
                    await websocket.send_text(
                        json.dumps({"event": "chunk", "content": piece})
                    )
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    allowed_sources = result.get("sources", [])
                    if service.is_insufficient_answer(answer_text):
                        filtered_sources = []
                        _chat_log(
                            "citation_guard skipped: insufficient_answer -> sources=[]",
                            trace_id,
                        )
                    else:
                        used_doc_ids = await service.infer_used_document_ids(
                            query=enriched_query,
                            answer_text=answer_text,
                            context_chunks=context_chunks,
                            allowed_sources=allowed_sources,
                            trace_id=trace_id,
                        )
                        filtered_sources = [
                            src
                            for src in allowed_sources
                            if src.get("document_id") in used_doc_ids
                        ]
                        _chat_log(
                            f"citation_guard used_doc_ids={used_doc_ids} filtered_sources={filtered_sources}",
                            trace_id,
                        )
                    await websocket.send_json(
                        {
                            "event": "done",
                            "sources": filtered_sources,
                            "confidence": confidence,
                        }
                    )
                    total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
                    _chat_log(
                        f"done tier=document confidence={confidence:.6f} "
                        f"sources={filtered_sources} elapsed_ms={total_elapsed_ms}",
                        trace_id,
                    )
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    _chat_log("send_done_failed: client_disconnected", trace_id)
                    break
                # === TIER 3: recommedation ===
            elif tier_source == "recommendation":
                print("floor 4: using recommendation layer")

                async for chunk in service.stream_response_from_recommendation(
                    user_id, session_id, enriched_query, message
                ):
                    await websocket.send_text(
                        json.dumps(
                            {
                                "event": "chunk",
                                "content": getattr(chunk, "content", str(chunk)),
                            }
                        )
                    )
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_json(
                        {"event": "done", "sources": [], "confidence": confidence}
                    )
                    total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
                    _chat_log(
                        f"done tier=recommendation confidence={confidence:.6f} sources=[] elapsed_ms={total_elapsed_ms}",
                        trace_id,
                    )
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    _chat_log("send_done_failed: client_disconnected", trace_id)
                    break

            elif tier_source == "document" or tier_source == "nope":
                print("floor 5: nope layer")
                async for chunk in service.stream_response_from_NA(
                    query=enriched_query,
                    context=context,
                    session_id=session_id,
                    user_id=user_id,
                    intent_id=0,
                    message=message,
                    current_audience_id=audience_id,
                    current_intent_id=intent_id_from_client,
                    query_embedding=result.get("query_embedding"),
                ):
                    await websocket.send_text(
                        json.dumps(
                            {
                                "event": "chunk",
                                "content": getattr(chunk, "content", str(chunk)),
                            }
                        )
                    )
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_json(
                        {"event": "done", "sources": [], "confidence": confidence}
                    )
                    total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
                    _chat_log(
                        f"done tier=nope confidence={confidence:.6f} "
                        f"sources=[] elapsed_ms={total_elapsed_ms}",
                        trace_id,
                    )
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    _chat_log("send_done_failed: client_disconnected", trace_id)
                    break

    except WebSocketDisconnect:
        # memory_manager.remove_memory(session_id)
        print("Client disconnected")


# ================== SSE STREAMING: THAY THẾ WEBSOCKET ==================


def _sse_event(data: dict) -> str:
    """Tạo một SSE event line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def stream_chat(
    request: Request,
    body: ChatStreamRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    SSE endpoint thay thế cho WebSocket /ws/chat.
    Client gửi POST với message, nhận về stream SSE gồm các event:
    - session: thông tin session
    - chunk: từng phần response
    - done: kết thúc, kèm sources và confidence
    - error: lỗi
    """

    confidence_threshold = float(os.getenv("CONFIDENCE_SCORE", 0.35))
    sse_service = TrainingService()

    message = body.message.strip()
    session_id = body.session_id
    user_id = body.user_id
    audience_id = body.audience_id
    intent_id_from_client = body.intent_id

    # Guest user/session có ID random không tồn tại trong DB
    # → reset None để service tự tạo mới (giống WS cũ)
    from app.models.entities import Users, ChatSession

    db_check = SessionLocal()
    try:
        if user_id:
            user_exists = (
                db_check.query(Users.user_id).filter(Users.user_id == user_id).first()
            )
            if not user_exists:
                user_id = None
        if session_id:
            session_exists = (
                db_check.query(ChatSession.chat_session_id)
                .filter(ChatSession.chat_session_id == session_id)
                .first()
            )
            if not session_exists:
                session_id = None
    finally:
        db_check.close()

    if not message:
        return StreamingResponse(
            iter([_sse_event({"event": "error", "message": "Message is empty"})]),
            media_type="text/event-stream",
        )

    # Tạo session nếu chưa có hoặc không tồn tại trong DB
    if not session_id:
        session_id = sse_service.create_chat_session(user_id, "chatbot")

    async def event_generator():
        top_k = os.getenv("TOP_K", 5)
        # Gửi session info
        yield _sse_event({"event": "session", "session_id": session_id})

        trace_id = uuid.uuid4().hex[:8]
        request_start = time.perf_counter()
        _chat_log(
            f"[SSE] incoming_message session_id={session_id} user_id={user_id} message_len={len(message)}",
            trace_id,
        )

        # --- enrich_query ---
        if audience_id == 4:
            top_k = 18
            enriched_query = await sse_service.enrich_query_tuyensinh(
                session_id, message
            )
        else:
            enriched_query = await sse_service.enrich_query(session_id, message)
        _chat_log(f"enriched_query={enriched_query}", trace_id)

        if not enriched_query:
            _chat_log("skip_rag: empty_enriched_query", trace_id)
            yield _sse_event(
                {
                    "event": "chunk",
                    "content": "Mình chưa rõ ý bạn lắm, bạn có thể nói rõ hơn được không?",
                }
            )
            yield _sse_event({"event": "done", "sources": [], "confidence": 0.0})
            return

        # --- Hybrid search ---
        try:
            hybrid_start = time.perf_counter()
            result = await sse_service.hybrid_search(
                audience_ids=audience_id,
                query=enriched_query,
                intent_id=intent_id_from_client,
                trace_id=trace_id,
            )
            hybrid_elapsed_ms = int((time.perf_counter() - hybrid_start) * 1000)
            _chat_log(f"hybrid_search_done elapsed_ms={hybrid_elapsed_ms}", trace_id)
        except Exception as e:
            _chat_log(
                f"hybrid_search_error type={type(e).__name__} message={e}", trace_id
            )
            yield _sse_event(
                {
                    "event": "chunk",
                    "content": "Hệ thống đang bận hoặc kho dữ liệu tạm thời không phản hồi. Bạn thử lại sau ít phút nhé.",
                }
            )
            yield _sse_event({"event": "done", "sources": [], "confidence": 0.0})
            return

        tier_source = result.get("response_source")
        confidence = result.get("confidence", 0.0)
        _chat_log(
            f"tier_source={tier_source} confidence={confidence:.6f} "
            f"sources={result.get('sources', [])}",
            trace_id,
        )
        print(f"audience id {audience_id}")
        # === TIER 1: training_qa ===
        if tier_source == "training_qa" and confidence > confidence_threshold:
            top = result["top_match"]
            q_text = top.payload.get("question_text")
            a_text = top.payload.get("answer_text")
            intent_id = top.payload.get("intent_id")
            relevance_ok = await sse_service.llm_relevance_check(
                enriched_query, q_text, a_text
            )
            _chat_log(f"training_qa_relevance={relevance_ok}", trace_id)

            if relevance_ok:
                # Check is_private cho training QA
                is_private = top.payload.get("is_private", False)
                if is_private and not current_user:
                    _chat_log("private_qa_blocked: user not logged in", trace_id)
                    yield _sse_event(
                        {
                            "event": "login_required",
                            "message": "Nội dung này yêu cầu đăng nhập để xem.",
                        }
                    )
                    yield _sse_event(
                        {"event": "done", "sources": [], "confidence": 0.0}
                    )
                    return

                async for chunk in sse_service.stream_response_from_qa(
                    enriched_query,
                    a_text,
                    session_id,
                    user_id,
                    intent_id,
                    message,
                ):
                    yield _sse_event(
                        {
                            "event": "chunk",
                            "content": getattr(chunk, "content", str(chunk)),
                        }
                    )
                yield _sse_event(
                    {"event": "done", "sources": [], "confidence": confidence}
                )
                total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
                _chat_log(
                    f"done tier=training_qa confidence={confidence:.6f} elapsed_ms={total_elapsed_ms}",
                    trace_id,
                )
                return
            else:
                # Fallback xuống document
                doc_results = await sse_service.search_documents(
                    enriched_query,
                    audience_ids=audience_id,
                    intent_id=intent_id_from_client,
                    top_k=top_k,
                    trace_id=trace_id,
                    stage="document_recheck_search",
                    query_embedding=result.get("query_embedding"),
                )
                result = sse_service.build_document_search_result(doc_results)
                confidence = result.get("confidence", 0.0)
                tier_source = "document"
                _chat_log(
                    f"qa_fallback_result confidence={confidence:.6f} sources={result.get('sources', [])}",
                    trace_id,
                )

        context_chunks = result["response"]
        intent_id = result["intent_id"]
        if sse_service.has_private_content(context_chunks) and not current_user:
            _chat_log("private_document_blocked: user not logged in", trace_id)
            yield _sse_event(
                {
                    "event": "login_required",
                    "message": "Nội dung này yêu cầu đăng nhập để xem.",
                }
            )
            yield _sse_event({"event": "done", "sources": [], "confidence": 0.0})
            return
        clean_context_chunks = []
        for r in context_chunks:
            chunk_text = r.payload.get("chunk_text", "")

            # Nếu người dùng đang hỏi cho năm 2026 hoặc hỏi chung chung về chỉ tiêu
            if "2026" in enriched_query:
                # Điều kiện trảm: Chunk chứa bảng thống kê năm cũ (có cả 2024 và 2025)
                if "Năm 2024" in chunk_text and "Năm 2025" in chunk_text:
                    # Ghi log để theo dõi hệ thống có chém nhầm không
                    _chat_log(
                        f"Filtered historical chunk: {chunk_text[:50]}...", trace_id
                    )
                    continue  # Bỏ qua chunk này, CHÉM!

            # Nếu chunk sạch (hoặc không dính điều kiện trên), thêm vào danh sách
            clean_context_chunks.append(r)
        context_chunks = clean_context_chunks
        context = "\n\n".join([r.payload.get("chunk_text", "") for r in context_chunks])
        _chat_log(
            f"context_precheck chunks={len(context_chunks)} chars={len(context)} intent_id={intent_id}",
            trace_id,
        )

        tier_check_start = time.perf_counter()
        tier_source = await sse_service.llm_document_recommendation_check(
            enriched_query, context
        )
        tier_check_elapsed_ms = int((time.perf_counter() - tier_check_start) * 1000)
        _chat_log(
            f"llm_document_recommendation_check result={tier_source} elapsed_ms={tier_check_elapsed_ms}",
            trace_id,
        )

        # === TIER 2: document ===
        if (
            tier_source == "document"
            and confidence > confidence_threshold
            and audience_id == 4
        ):
            answer_text = ""
            print(f"SCORE BEFORE DOC: {confidence}")
            async for chunk in sse_service.stream_response_from_context_tuyensinh(
                enriched_query,
                context,
                session_id,
                user_id,
                intent_id,
                message,
                query_embedding=result.get("query_embedding"),
                current_audience_id=audience_id,
                current_intent_id=intent_id_from_client,
                confidence=confidence,
            ):
                piece = getattr(chunk, "content", str(chunk))
                answer_text += piece
                yield _sse_event({"event": "chunk", "content": piece})

            # Citation guard
            allowed_sources = result.get("sources", [])
            if sse_service.is_insufficient_answer(answer_text):
                filtered_sources = []
                _chat_log(
                    "citation_guard skipped: insufficient_answer -> sources=[]",
                    trace_id,
                )
            else:
                used_doc_ids = await sse_service.infer_used_document_ids(
                    query=enriched_query,
                    answer_text=answer_text,
                    context_chunks=context_chunks,
                    allowed_sources=allowed_sources,
                    trace_id=trace_id,
                )
                filtered_sources = [
                    src
                    for src in allowed_sources
                    if src.get("document_id") in used_doc_ids
                ]
                _chat_log(
                    f"citation_guard used_doc_ids={used_doc_ids} filtered_sources={filtered_sources}",
                    trace_id,
                )

            yield _sse_event(
                {
                    "event": "done",
                    "sources": filtered_sources,
                    "confidence": confidence,
                }
            )
            total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
            _chat_log(
                f"done tier=document confidence={confidence:.6f} "
                f"sources={filtered_sources} elapsed_ms={total_elapsed_ms}",
                trace_id,
            )
            return
        elif tier_source == "document" and confidence > confidence_threshold:
            answer_text = ""
            print(f"SCORE BEFORE DOC: {confidence}")
            async for chunk in sse_service.stream_response_from_context(
                enriched_query,
                context,
                session_id,
                user_id,
                intent_id,
                message,
                query_embedding=result.get("query_embedding"),
                current_audience_id=audience_id,
                current_intent_id=intent_id_from_client,
                confidence=confidence,
            ):
                piece = getattr(chunk, "content", str(chunk))
                answer_text += piece
                yield _sse_event({"event": "chunk", "content": piece})

            # Citation guard
            allowed_sources = result.get("sources", [])
            if sse_service.is_insufficient_answer(answer_text):
                filtered_sources = []
                _chat_log(
                    "citation_guard skipped: insufficient_answer -> sources=[]",
                    trace_id,
                )
            else:
                used_doc_ids = await sse_service.infer_used_document_ids(
                    query=enriched_query,
                    answer_text=answer_text,
                    context_chunks=context_chunks,
                    allowed_sources=allowed_sources,
                    trace_id=trace_id,
                )
                filtered_sources = [
                    src
                    for src in allowed_sources
                    if src.get("document_id") in used_doc_ids
                ]
                _chat_log(
                    f"citation_guard used_doc_ids={used_doc_ids} filtered_sources={filtered_sources}",
                    trace_id,
                )

            yield _sse_event(
                {
                    "event": "done",
                    "sources": filtered_sources,
                    "confidence": confidence,
                }
            )
            total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
            _chat_log(
                f"done tier=document confidence={confidence:.6f} "
                f"sources={filtered_sources} elapsed_ms={total_elapsed_ms}",
                trace_id,
            )
            return
        # === TIER 3: recommendation ===
        # elif tier_source == "recommendation":
        #     async for chunk in sse_service.stream_response_from_recommendation(
        #         user_id, session_id, enriched_query, message
        #     ):
        #         yield _sse_event(
        #             {
        #                 "event": "chunk",
        #                 "content": getattr(chunk, "content", str(chunk)),
        #             }
        #         )
        #     yield _sse_event({"event": "done", "sources": [], "confidence": confidence})
        #     total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
        #     _chat_log(
        #         f"done tier=recommendation confidence={confidence:.6f} sources=[] elapsed_ms={total_elapsed_ms}",
        #         trace_id,
        #     )
        #     return

        # === TIER 4: nope / low-confidence document ===
        elif tier_source == "document" or tier_source == "nope":
            async for chunk in sse_service.stream_response_from_NA(
                query=enriched_query,
                context=context,
                session_id=session_id,
                user_id=user_id,
                intent_id=0,
                message=message,
                current_audience_id=audience_id,
                current_intent_id=intent_id_from_client,
                query_embedding=result.get("query_embedding"),
            ):
                yield _sse_event(
                    {
                        "event": "chunk",
                        "content": getattr(chunk, "content", str(chunk)),
                    }
                )
            yield _sse_event({"event": "done", "sources": [], "confidence": confidence})
            total_elapsed_ms = int((time.perf_counter() - request_start) * 1000)
            _chat_log(
                f"done tier=nope confidence={confidence:.6f} "
                f"sources=[] elapsed_ms={total_elapsed_ms}",
                trace_id,
            )
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ================== REST API: SESSION ==================


# Tạo 1 phiên chat session mới
@router.post("/session/create")
def api_create_chat_session(user_id: int, session_type: str):
    try:
        session_id = service.create_chat_session(
            user_id=user_id, session_type=session_type
        )
        return {"session_id": session_id, "message": "Session created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Lấy lịch sử chat theo session ID
@router.get("/session/{session_id}/history")
def api_get_session_history(session_id: int, limit: int = 50):
    try:
        history = service.get_session_history(session_id, limit)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Lấy tất cả session của user
@router.get("/user/{user_id}/sessions")
def api_get_user_sessions(user_id: int):
    try:
        sessions = service.get_user_sessions(user_id)
        return {"user_id": user_id, "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Xóa 1 session chat
@router.delete("/session/{session_id}")
def api_delete_chat_session(session_id: int, user_id: int | None = None):
    """
    - Nếu truyền user_id: chỉ cho xóa session thuộc user đó
    - Nếu không truyền user_id: xóa theo session_id (guest session)
    """
    try:
        deleted = service.delete_chat_session(session_id=session_id, user_id=user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
