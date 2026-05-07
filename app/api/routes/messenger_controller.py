from fastapi import APIRouter, Query, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import logging
import os

from app.services.facebook_service import facebook_service
from app.services.training_service import TrainingService
from app.models.database import get_db
from app.models.entities import TargetAudience

router = APIRouter()
logger = logging.getLogger(__name__)

service = TrainingService()

# In-memory cache: sender_psid -> session info
# Production nên dùng Redis hoặc DB, nhưng đơn giản dùng dict cho POC
# Structure: {sender_psid: {"session_id": int, "audience_id": int | None}}
_messenger_sessions: dict[str, dict] = {}


def get_or_create_session(sender_psid: str) -> dict:
    """Lấy hoặc tạo session info cho messenger user."""
    if sender_psid in _messenger_sessions:
        return _messenger_sessions[sender_psid]
    session_id = service.create_chat_session(user_id=None, session_type="messenger")
    info = {"session_id": session_id, "audience_id": None}
    _messenger_sessions[sender_psid] = info
    return info


def get_audiences(db=None) -> list[dict]:
    """Lấy danh sách audience từ DB."""
    try:
        from app.models.database import SessionLocal
        if db is None:
            db = SessionLocal()
        rows = db.query(TargetAudience).all()
        return [{"id": r.id, "name": r.name} for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch audiences: {e}")
        return []


def build_audience_quick_replies() -> list[dict]:
    """Build quick replies cho audience selection."""
    audiences = get_audiences()
    quick_replies = []
    for aud in audiences:
        quick_replies.append({
            "content_type": "text",
            "title": aud["name"],
            "payload": f"AUDIENCE:{aud['id']}",
        })
    return quick_replies


class MessengerWebhookPayload(BaseModel):
    object: str
    entry: list


@router.get("", response_class=PlainTextResponse)
async def verify_webhook(
    mode: str = Query(..., alias="hub.mode"),
    token: str = Query(..., alias="hub.verify_token"),
    challenge: str = Query(..., alias="hub.challenge"),
):
    """
    Facebook gọi GET này để verify webhook URL khi setup.
    Trả về challenge nếu verify_token khớp.
    """
    result = facebook_service.verify_webhook(mode, token, challenge)
    if result:
        logger.info("Webhook verified successfully.")
        return PlainTextResponse(content=result)
    logger.warning("Webhook verification failed.")
    return PlainTextResponse(content="Verification failed", status_code=403)


@router.post("")
async def handle_webhook(payload: MessengerWebhookPayload):
    """
    Nhận message events từ Facebook Messenger.
    Xử lý: text message -> chatbot RAG -> reply.
    Hỗ trợ audience selection qua quick reply payloads.
    """
    if payload.object != "page":
        return {"status": "ignored"}

    for entry in payload.entry:
        for event in entry.get("messaging", []):
            sender_psid = event.get("sender", {}).get("id")
            recipient_psid = event.get("recipient", {}).get("id")
            message = event.get("message", {})
            timestamp = event.get("timestamp")

            logger.info(
                f" Messenger event: sender={sender_psid}, recipient={recipient_psid}, "
                f"timestamp={timestamp}"
            )

            if not sender_psid:
                continue

            # Xử lý postback (persistent menu click)
            postback = event.get("postback", {})
            if postback:
                payload_data = postback.get("payload", "")

                # Handle GET_STARTED
                if payload_data == "GET_STARTED":
                    quick_replies = build_audience_quick_replies()
                    if quick_replies:
                        facebook_service.send_quick_replies(
                            sender_psid,
                            "👋 Xin chào! Mình là trợ lý tư vấn tuyển sinh.\n\n"
                            "Bạn là **đối tượng** nào? (Chọn 1 trong các options bên dưới):",
                            quick_replies,
                        )
                    else:
                        facebook_service.send_text_message(
                            sender_psid,
                            "👋 Xin chào! Mình là trợ lý tư vấn tuyển sinh.\n\n"
                            "Bạn vui lòng nhắn 'sinh viên', 'phụ huynh', 'viên chức', "
                            "hoặc 'tuyển sinh' để mình tư vấn đúng nhé!"
                        )
                    return {"status": "ok"}

                # Handle RESET_SESSION
                if payload_data == "RESET_SESSION":
                    if sender_psid in _messenger_sessions:
                        del _messenger_sessions[sender_psid]
                    quick_replies = build_audience_quick_replies()
                    if quick_replies:
                        facebook_service.send_quick_replies(
                            sender_psid,
                            "🔄 Đã bắt đầu lại!\n\nBạn là **đối tượng** nào?",
                            quick_replies,
                        )
                    else:
                        facebook_service.send_text_message(
                            sender_psid,
                            "🔄 Đã bắt đầu lại! Bạn vui lòng nhắn 'sinh viên', "
                            "'phụ huynh', 'viên chức', hoặc 'tuyển sinh' nhé!"
                        )
                    return {"status": "ok"}

                if payload_data.startswith("AUDIENCE:"):
                    try:
                        audience_id = int(payload_data.split(":")[1])
                        info = get_or_create_session(sender_psid)
                        info["audience_id"] = audience_id
                        _messenger_sessions[sender_psid] = info

                        audiences = get_audiences()
                        selected_name = next(
                            (a["name"] for a in audiences if a["id"] == audience_id),
                            f"Audience #{audience_id}"
                        )
                        facebook_service.send_text_message(
                            sender_psid,
                            f"✓ Đã chọn: **{selected_name}**\n\n"
                            f"Bây giờ bạn có thể hỏi về chủ đề liên quan nhé!"
                        )
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid audience payload: {payload_data}")
                    return {"status": "ok"}

            if not message:
                continue

            # Bỏ qua message có "is_echo" (bot tự gửi)
            if message.get("is_echo"):
                continue

            # Xử lý quick reply payload (khi user bấm quick reply button)
            quick_reply_payload = message.get("quick_reply", {}).get("payload", "")
            if quick_reply_payload.startswith("AUDIENCE:"):
                try:
                    audience_id = int(quick_reply_payload.split(":")[1])
                    info = get_or_create_session(sender_psid)
                    info["audience_id"] = audience_id
                    _messenger_sessions[sender_psid] = info

                    audiences = get_audiences()
                    selected_name = next(
                        (a["name"] for a in audiences if a["id"] == audience_id),
                        f"Audience #{audience_id}"
                    )
                    facebook_service.send_text_message(
                        sender_psid,
                        f"✓ Đã chọn: **{selected_name}**\n\n"
                        f"Bây giờ bạn có thể hỏi về chủ đề liên quan nhé!"
                    )
                except (ValueError, IndexError):
                    logger.warning(f"Invalid quick reply payload: {quick_reply_payload}")
                    facebook_service.send_text_message(
                        sender_psid,
                        "Mình chưa hiểu lựa chọn của bạn. Bạn thử chọn lại nhé!"
                    )
                continue

            # Bỏ qua message có attachments (chỉ xử lý text)
            if message.get("attachments"):
                facebook_service.send_text_message(
                    sender_psid,
                    "Hiện tại mình chỉ hỗ trợ tin nhắn văn bản thôi nhé. "
                    "Bạn cứ gửi câu hỏi bằng text, mình sẽ trả lời ngay!",
                )
                continue

            text = message.get("text", "").strip()
            if not text:
                continue

            # Bật typing indicator trong khi xử lý
            facebook_service.send_typing_on(sender_psid)

            try:
                info = get_or_create_session(sender_psid)
                session_id = info["session_id"]
                audience_id = info.get("audience_id")

                # Lần đầu chưa chọn audience -> hỏi chọn audience
                if audience_id is None:
                    quick_replies = build_audience_quick_replies()
                    if quick_replies:
                        facebook_service.send_quick_replies(
                            sender_psid,
                            "👋 Xin chào! Mình là trợ lý tư vấn tuyển sinh.\n\n"
                            "Bạn là **đối tượng** nào? (Chọn 1 trong các options bên dưới):",
                            quick_replies,
                        )
                    else:
                        facebook_service.send_text_message(
                            sender_psid,
                            "👋 Xin chào! Mình là trợ lý tư vấn tuyển sinh.\n\n"
                            "Bạn vui lòng nhắn 'sinh viên', 'phụ huynh', 'viên chức', "
                            "hoặc 'tuyển sinh' để mình tư vấn đúng nhé!"
                        )
                    facebook_service.send_typing_off(sender_psid)
                    continue

                # Enrich query
                enriched_query = await service.enrich_query(session_id, text)
                if not enriched_query:
                    reply = (
                        "Mình chưa rõ ý bạn lắm, bạn có thể nói rõ hơn được không?"
                    )
                    facebook_service.send_text_message(sender_psid, facebook_service.strip_markdown(reply))
                    facebook_service.send_typing_off(sender_psid)
                    continue

                # Hybrid search - đợi full response (không stream như WS)
                top_k = int(os.getenv("TOP_K", 5))
                confidence_threshold = float(os.getenv("CONFIDENCE_SCORE", 0.35))

                result = await service.hybrid_search(
                    audience_ids=audience_id,
                    query=enriched_query,
                    intent_id=None,
                    trace_id="fb",
                )
                tier_source = result.get("response_source", "nope")
                confidence = result.get("confidence", 0.0)
                query_embedding = result.get("query_embedding")

                logger.info(
                    f"Messenger RAG: tier={tier_source}, confidence={confidence:.4f}, audience_id={audience_id}"
                )

                # === TIER 1: training_qa với relevance check ===
                if tier_source == "training_qa" and confidence >= confidence_threshold:
                    top = result["top_match"]
                    q_text = top.payload.get("question_text")
                    a_text = top.payload.get("answer_text")
                    intent_id = top.payload.get("intent_id")

                    relevance_ok = await service.llm_relevance_check(
                        enriched_query, q_text, a_text
                    )
                    logger.info(f"Messenger training_qa_relevance={relevance_ok}")

                    if relevance_ok:
                        reply_chunks = []
                        async for chunk in service.stream_response_from_qa(
                            enriched_query,
                            a_text,
                            session_id,
                            None,
                            intent_id,
                            text,
                        ):
                            reply_chunks.append(getattr(chunk, "content", str(chunk)))
                        reply = "".join(reply_chunks)
                        facebook_service.send_text_message(sender_psid, facebook_service.strip_markdown(reply))
                        facebook_service.send_typing_off(sender_psid)
                        continue
                    else:
                        # QA not relevant → fallback xuống document
                        logger.info("Messenger: QA not relevant → fallback to document")
                        doc_results = await service.search_documents(
                            enriched_query,
                            audience_ids=audience_id,
                            intent_id=None,
                            top_k=top_k,
                            trace_id="fb",
                            stage="messenger_document_recheck",
                            query_embedding=query_embedding,
                        )
                        result = service.build_document_search_result(doc_results)
                        confidence = result.get("confidence", 0.0)
                        tier_source = "document"
                        query_embedding = result.get("query_embedding")

                # Build context từ document chunks
                context_chunks = result.get("response", [])
                context = "\n\n".join(
                    [r.payload.get("chunk_text", "") for r in context_chunks]
                )
                intent_id = result.get("intent_id")

                logger.info(
                    f"Messenger context: chunks={len(context_chunks)} chars={len(context)}"
                )

                # LLM check xem nên dùng document hay recommendation hay nope
                tier_check = await service.llm_document_recommendation_check(
                    enriched_query, context
                )
                logger.info(f"Messenger tier_check={tier_check}")
                tier_source = tier_check

                # === TIER 2: document ===
                if tier_source == "document" and confidence >= confidence_threshold:
                    reply_chunks = []
                    async for chunk in service.stream_response_from_context(
                        enriched_query,
                        context,
                        session_id,
                        None,
                        intent_id,
                        text,
                    ):
                        piece = getattr(chunk, "content", str(chunk))
                        reply_chunks.append(piece)
                    reply = "".join(reply_chunks)

                    # Citation guard
                    allowed_sources = result.get("sources", [])
                    if not service.is_insufficient_answer(reply):
                        used_doc_ids = await service.infer_used_document_ids(
                            query=enriched_query,
                            answer_text=reply,
                            context_chunks=context_chunks,
                            allowed_sources=allowed_sources,
                            trace_id="fb",
                        )
                        filtered_sources = [
                            src
                            for src in allowed_sources
                            if src.get("document_id") in used_doc_ids
                        ]
                        logger.info(f"Messenger citation_guard used_docs={used_doc_ids}")
                    else:
                        filtered_sources = []
                        logger.info("Messenger citation_guard skipped: insufficient_answer")

                    facebook_service.send_text_message(sender_psid, facebook_service.strip_markdown(reply))
                    facebook_service.send_typing_off(sender_psid)
                    continue

                # === TIER 3: recommendation ===
                if tier_source == "recommendation":
                    reply_chunks = []
                    async for chunk in service.stream_response_from_recommendation(
                        None, session_id, enriched_query, text
                    ):
                        reply_chunks.append(getattr(chunk, "content", str(chunk)))
                    reply = "".join(reply_chunks)
                    facebook_service.send_text_message(sender_psid, facebook_service.strip_markdown(reply))
                    facebook_service.send_typing_off(sender_psid)
                    continue

                # === TIER 4: nope ===
                reply_chunks = []
                async for chunk in service.stream_response_from_NA(
                    query=enriched_query,
                    context=context,
                    session_id=session_id,
                    user_id=None,
                    intent_id=0,
                    message=text,
                    current_audience_id=audience_id,
                    current_intent_id=None,
                    query_embedding=query_embedding,
                ):
                    reply_chunks.append(getattr(chunk, "content", str(chunk)))
                reply = "".join(reply_chunks)

                if not reply or not reply.strip():
                    reply = (
                        "Xin lỗi, mình chưa có câu trả lời phù hợp cho câu hỏi này. "
                        "Bạn thử hỏi theo cách khác nhé!"
                    )

                facebook_service.send_text_message(sender_psid, facebook_service.strip_markdown(reply))

            except Exception as e:
                logger.error(f"Error handling messenger message: {e}")
                facebook_service.send_text_message(
                    sender_psid,
                    "Xin lỗi, hệ thống đang bận. Bạn thử lại sau ít phút nhé.",
                )
            finally:
                facebook_service.send_typing_off(sender_psid)

    return {"status": "ok"}


@router.post("/setup-menu")
async def setup_persistent_menu():
    """
    Gọi endpoint này 1 lần để setup Persistent Menu trên Messenger.
    Menu cho phép user đổi audience bất kỳ lúc nào.
    """
    from app.core.config import settings

    menu_items = [
        {
            "type": "postback",
            "title": "👨‍🎓 Sinh viên",
            "payload": "AUDIENCE:1",
        },
        {
            "type": "postback",
            "title": "👨‍👩‍👧 Phụ huynh",
            "payload": "AUDIENCE:2",
        },
        {
            "type": "postback",
            "title": "👔 Viên chức",
            "payload": "AUDIENCE:3",
        },
        {
            "type": "postback",
            "title": "🔍 Tuyển sinh",
            "payload": "AUDIENCE:4",
        },
        {
            "type": "postback",
            "title": "🔄 Bắt đầu lại",
            "payload": "RESET_SESSION",
        },
    ]

    import requests

    # Xóa menu cũ trước (nếu có)
    delete_url = "https://graph.facebook.com/v18.0/me/messenger_profile"
    delete_payload = {
        "access_token": settings.FACEBOOK_PAGE_ACCESS_TOKEN,
        "fields": "persistent_menu",
    }
    requests.delete(delete_url, json=delete_payload, timeout=10)

    # Setup Get Started button (bắt buộc trước persistent menu)
    get_started_url = "https://graph.facebook.com/v18.0/me/messenger_profile"
    get_started_payload = {
        "access_token": settings.FACEBOOK_PAGE_ACCESS_TOKEN,
        "get_started": {"payload": "GET_STARTED"},
    }
    resp_gs = requests.post(get_started_url, json=get_started_payload, timeout=10)
    logger.info(f"Get Started setup: {resp_gs.json()}")

    # Setup Persistent Menu
    url = "https://graph.facebook.com/v18.0/me/messenger_profile"
    payload = {
        "access_token": settings.FACEBOOK_PAGE_ACCESS_TOKEN,
        "persistent_menu": [
            {
                "locale": "default",
                "composer_input_disabled": False,
                "call_to_actions": menu_items,
            }
        ],
    }
    resp = requests.post(url, json=payload)
    result = resp.json()

    if resp.status_code == 200:
        logger.info("Persistent menu setup successful.")
        return {"status": "ok", "message": "Persistent menu created successfully."}
    else:
        logger.error(f"Persistent menu setup failed: {result}")
        return {"status": "error", "detail": result}
