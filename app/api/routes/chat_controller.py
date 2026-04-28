from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
import asyncio
import json
import time
import uuid
from app.models.database import SessionLocal
from app.services import training_service
from app.services.training_service import TrainingService


router = APIRouter()
# Tạo 1 instance dùng chung
service = TrainingService()


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
                result = service.hybrid_search(
                    audience_id,
                    enriched_query,
                    intent_id_from_client,
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
            if tier_source == "training_qa" and confidence > 0.35:
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
                    doc_results = service.search_documents(
                        enriched_query,
                        audience_ids=audience_id,
                        intent_id=intent_id_from_client,
                        top_k=5,
                        trace_id=trace_id,
                        stage="document_recheck_search",
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
            if tier_source == "document":
                doc_results = service.search_documents(
                    enriched_query,
                    audience_ids=audience_id,
                    intent_id=intent_id_from_client,
                    top_k=5,
                    trace_id=trace_id,
                    stage="document_recheck_search",
                )
                result = service.build_document_search_result(doc_results)
                confidence = result.get("confidence", 0.0)
                context_chunks = result["response"]
                intent_id = result["intent_id"]
                context = "\n\n".join(
                    [r.payload.get("chunk_text", "") for r in context_chunks]
                )
                _chat_log(
                    f"document_recheck_result confidence={confidence:.6f} "
                    f"chunks={len(context_chunks)} chars={len(context)} sources={result.get('sources', [])}",
                    trace_id,
                )
                print("Context:" + context)
                print("Confidence of document:")
                print(confidence)
            print("SOURCE NAME: " + tier_source)
            # === TIER 2: document-only (no QA match) ===
            if tier_source == "document" and confidence >= 0.35:
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
                    enriched_query,
                    context,
                    session_id,
                    user_id,
                    0,
                    message,
                    audience_id,
                    intent_id_from_client,
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
