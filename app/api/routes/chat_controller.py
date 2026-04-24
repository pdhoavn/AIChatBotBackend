from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
import asyncio
import json
from app.models.database import SessionLocal
from app.services import training_service
from app.services.training_service import TrainingService


router = APIRouter()
# Tạo 1 instance dùng chung
service = TrainingService()
#thêm 3 tầng check chat
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
        await websocket.send_json({
            "event": "session_created",
            "session_id": session_id
        })

    await websocket.send_json({"event": "go", "sources": [], "confidence": 1.0})
 
    try:
        while True:
            # Nhận tin nhắn từ client
            raw_data = await websocket.receive_json()
            message = raw_data.get("message", "").strip()
            if not message:
                continue

             # enrich_query — tạo truy vấn "đầy đủ" dựa vào hội thoại cũ
            enriched_query = await service.enrich_query(session_id, message)
            print(f"👉 enriched_query: {enriched_query}")

             # Nếu enrich_query rỗng, nghĩa là user nói lan man → không cần RAG
            if not enriched_query:
                await websocket.send_json({
                    "event": "chunk",
                    "content": "Mình chưa rõ ý bạn lắm, bạn có thể nói rõ hơn được không?"
                })
                await websocket.send_json({"event": "done", "sources": [], "confidence": 0.0})
                continue


            # Tìm context liên quan
            # doc_results = TrainingService.search_documents(message, top_k=5)
           
            # Hybrid search (cả training QA và document)
            try:
                result = service.hybrid_search(enriched_query)
            except Exception as e:
                print(f"Hybrid search error: {e}")
                await websocket.send_json({
                    "event": "chunk",
                    "content": "Hệ thống đang bận hoặc kho dữ liệu tạm thời không phản hồi. Bạn thử lại sau ít phút nhé."
                })
                await websocket.send_json({"event": "done", "sources": [], "confidence": 0.0})
                continue
            tier_source = result.get("response_source")
            confidence = result.get("confidence", 0.0)
            print(f"👉 tier_source: {tier_source}")
            print(f"👉 confidence: {confidence}")
            # === TIER 1: training_qa - score > 0.8 ===
            if tier_source == "training_qa" and confidence > 0.5:
                print("floor 1")
                top = result["top_match"]
                q_text = top.payload.get("question_text")
                a_text = top.payload.get("answer_text")
                intent_id = top.payload.get("intent_id")
                relevance_ok = await service.llm_relevance_check(enriched_query, q_text, a_text)

                if relevance_ok:
                    print("floor 1: training QA valid")
                    async for chunk in service.stream_response_from_qa(enriched_query, a_text, session_id, user_id, intent_id, message):
                        await websocket.send_text(json.dumps({
                            "event": "chunk",
                            "content": getattr(chunk, "content", str(chunk))
                        }))
                    await websocket.send_json({
                        "event": "done",
                        "sources": [q_text],
                        "confidence": confidence
                    })
                    continue
                else:
                    print("QA not relevant → fallback xuống document")
                    # Chạy document search lại
                    doc_results = service.search_documents(enriched_query, top_k=5)
                    result = {
                        "response": doc_results,
                        "intent_id": doc_results[0].payload.get("intent_id"),
                        "response_source": "document",
                        "confidence": doc_results[0].score if doc_results else 0.0,
                        "sources": [r.payload.get("document_id") for r in doc_results]
                    }
                    tier_source = "document"
                    
            context_chunks = result["response"]
            intent_id = result["intent_id"]
            context = "\n\n".join([
                r.payload.get("chunk_text", "") for r in context_chunks
            ])
            
            tier_source = await service.llm_document_recommendation_check(enriched_query, context)
            if(tier_source == "document"):
                doc_results = service.search_documents(enriched_query, top_k=5)
                result = {
                    "response": doc_results,
                    "intent_id": doc_results[0].payload.get("intent_id"),
                    "response_source": "document",
                    "confidence": doc_results[0].score if doc_results else 0.0,
                    "sources": [r.payload.get("document_id") for r in doc_results]
                }
                confidence = doc_results[0].score if doc_results else 0.0
                print("Context:" + context)
                print("Confidence of document:")
                print(confidence)
            print("SOURCE NAME: " + tier_source)
            # === TIER 2: document-only (no QA match) ===
            if tier_source == "document" and confidence >= 0.5:
                print("🔍 floor 3: using document context")
                
                async for chunk in service.stream_response_from_context(
                    enriched_query, context, session_id, user_id, intent_id, message
                ):
                    await websocket.send_text(json.dumps({
                        "event": "chunk",
                        "content": getattr(chunk, "content", str(chunk))
                    }))
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_json({
                        "event": "done",
                        "sources": result.get("sources", []),
                        "confidence": confidence
                    })
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    break
                # === TIER 3: recommedation ===
            elif tier_source == "recommendation":
                print("floor 4: using recommendation layer")
                   
                async for chunk in service.stream_response_from_recommendation(
                    user_id, session_id, enriched_query, message
                ):
                    await websocket.send_text(json.dumps({
                        "event": "chunk",
                        "content": getattr(chunk, "content", str(chunk))
                    }))
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_json({
                        "event": "done",
                        "sources": result.get("sources", []),
                        "confidence": confidence
                    })
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
                    break

            elif tier_source == "document" or tier_source == "nope":
                print("floor 5: nope layer")
                async for chunk in service.stream_response_from_NA(
                    enriched_query, context, session_id, user_id, 0, message
                ):
                    await websocket.send_text(json.dumps({
                        "event": "chunk",
                        "content": getattr(chunk, "content", str(chunk))
                    }))
                    # Gửi tín hiệu kết thúc khi hoàn tất
                try:
                    await websocket.send_json({
                        "event": "done",
                        "sources": result.get("sources", []),
                        "confidence": confidence
                    })
                    continue
                except Exception:
                    print("Không thể gửi event done vì client đã ngắt.")
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
            user_id=user_id,
            session_type=session_type
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
