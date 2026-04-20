import asyncio
import json
from fastapi import APIRouter, Request, WebSocket, Response
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from starlette.websockets import WebSocketDisconnect
from app.models.entities import ChatInteraction, LiveChatQueue
from app.services.livechat_service import LiveChatService
from app.models.database import SessionLocal

# Create a singleton instance of the LiveChatService
live_chat_service = LiveChatService()

router = APIRouter(prefix="/livechat", tags=["Live Chat"])

@router.post("/live-chat/join_queue")
async def join_queue(customer_id: int):
    return await live_chat_service.customer_join_queue(customer_id)

#xem trang thai 
@router.get("/customer/queue/status/{customer_id}")
async def get_my_queue_status(customer_id: int):
    return live_chat_service.get_my_status(customer_id)

#customer cancel queue request
@router.post("/customer/cancel_queue")
async def cancel_queue_request(customer_id: int):
    """
    Customer cancels their pending queue request.
    Only works when status is 'waiting' (before officer acceptance).
    """
    return await live_chat_service.customer_cancel_queue(customer_id)

@router.delete("live-chat/queue/{queue_id}")
async def delete_queue(queue_id: int):
    return {"deleted": live_chat_service.delete_queue_item(queue_id)}

#xem tin nhan trong session live chat
@router.get("/session/{session_id}/messages")
async def get_messages(session_id: int):
    return live_chat_service.get_messages(session_id)

# @router.get("/sessions/user/{user_id}")
# async def list_sessions(user_id: int):
#     db = SessionLocal()
#     sessions = db.query(ParticipateChatSession)\
#         .filter_by(user_id=user_id)\
#         .all()
#     return sessions

# @router.get("/session/{session_id}")
# async def get_session(session_id: int):
#     db = SessionLocal()
#     return db.query(ChatSession).filter_by(chat_session_id=session_id).first()

#admission official xem danh sach cac customer co trong hang doi
@router.get("/admission_official/queue/list/{official_id}")
async def get_queue(official_id: int):
    return live_chat_service.get_queue_list(official_id)

#admission official xem danh sach cac session dang hoat dong
@router.get("/admission_official/active_sessions/{official_id}")
async def get_active_sessions(official_id: int):
    return await live_chat_service.get_active_sessions(official_id)

#admission offcial accept 1 queue(1 customer)
@router.post("/admission_official/accept")
async def accept_request(official_id: int, queue_id: int):
    print(f"\n[API] /admission_official/accept called with official_id={official_id}, queue_id={queue_id}")
    result = await live_chat_service.official_accept(official_id, queue_id)
    
    # Result is now always a dict
    if isinstance(result, dict):
        if "error" in result:
            print(f"[API] ⚠️ Returning error: {result}")
        else:
            print(f"[API] ✅ Returning success with session_id={result.get('session_id')}")
    
    return result


@router.post("/admission_official/reject")
async def reject_request(official_id: int, queue_id: int, reason: str):
    return await live_chat_service.official_reject(official_id, queue_id, reason)

#ket thuc session live chat
@router.post("/live-chat/end")
async def end_session(session_id: int, ended_by: int):
    return await live_chat_service.end_session(session_id, ended_by)

# Debug endpoint: Get SSE connection counts
@router.get("/debug/sse-connections")
async def debug_sse_connections(customer_id: int = None, official_id: int = None):
    """
    Debug endpoint to check SSE connection counts.
    Use customer_id or official_id to check specific user, or omit both to see all.
    """
    if customer_id:
        count = live_chat_service.get_sse_connection_count(customer_id=customer_id)
        return {"customer_id": customer_id, "connection_count": count}
    elif official_id:
        count = live_chat_service.get_sse_connection_count(official_id=official_id)
        return {"official_id": official_id, "connection_count": count}
    else:
        return live_chat_service.get_sse_connection_count()

# Reset/Clear SSE connections for a specific customer (emergency cleanup)
@router.post("/debug/reset-sse-customer/{customer_id}")
async def reset_customer_sse(customer_id: int):
    """
    Emergency endpoint to forcefully clear all SSE connections for a customer.
    Use this if SSE connections are stuck/duplicated.
    """
    if customer_id in live_chat_service.sse_customers:
        count = len(live_chat_service.sse_customers[customer_id])
        live_chat_service.sse_customers[customer_id].clear()
        del live_chat_service.sse_customers[customer_id]
        return {
            "success": True,
            "message": f"Cleared {count} SSE connection(s) for customer {customer_id}"
        }
    return {
        "success": False,
        "message": f"No SSE connections found for customer {customer_id}"
    }

#Server-Sent Events: sẽ trả về bên phía customer cho mấy cái như thông báo tin nhắn tới hay thông báo hàng đợi của mình được chấp nhận hay render theo thời gian thực
# { "event": "queue_updated",  "data": { "queue_id": 5 } }
# { "event": "accepted",  "data": { "queue_id": 2 } }

# Handle CORS preflight for customer SSE
@router.options("/sse/customer/{customer_id}")
async def customer_sse_preflight(customer_id: int):
    """Handle CORS preflight request for customer SSE endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )

@router.get("/sse/customer/{customer_id}")
async def customer_sse(request: Request, customer_id: int):
    print(f"[SSE] New customer connection request for customer_id={customer_id}")
    
    queue = asyncio.Queue()
    connection_active = True

    async def send_event(data: dict):
        if connection_active:
            try:
                await queue.put(data)
            except Exception as e:
                print(f"[SSE] Failed to queue event for customer {customer_id}: {e}")
                raise

    # Đăng ký callback vào service
    live_chat_service.register_customer_sse(customer_id, send_event)

    async def event_stream():
        nonlocal connection_active
        try:
            # Send initial connection event
            print(f"[SSE] Customer {customer_id} connection established")
            yield f"data: {json.dumps({'event': 'connected', 'message': 'SSE connection established', 'customer_id': customer_id})}\n\n"
            
            ping_counter = 0
            while True:
                try:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        print(f"[SSE] Customer {customer_id} disconnected (client closed)")
                        break
                    
                    # Wait for data with timeout to allow disconnect checking
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Send the actual event
                    event_data = {
                        "event": data.get("event", "update"),
                        "data": data
                    }
                    print(f"[SSE] Sending event to customer {customer_id}: {data.get('event')}")
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send periodic heartbeat to keep connection alive (every 10 pings = 10 seconds, log once)
                    ping_counter += 1
                    if ping_counter % 10 == 0:
                        print(f"[SSE] Customer {customer_id} connection alive (ping #{ping_counter})")
                    yield f"data: {json.dumps({'event': 'ping', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                    continue
                    
        except GeneratorExit:
            print(f"[SSE] Customer {customer_id} connection closed (GeneratorExit)")
        except Exception as e:
            print(f"[SSE] Customer {customer_id} error: {e}")
        finally:
            connection_active = False
            print(f"[SSE] Unregistering customer {customer_id} SSE connection")
            live_chat_service.unregister_customer_sse(customer_id, send_event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"
        }
    )

#Server-Sent Events: sẽ trả về cho bên phía admission official cho mấy cái như thông báo tin nhắn tới hay thông báo hàng đợi của mình được chấp nhận hay render theo thời gian thực
# Handle CORS preflight for SSE
@router.options("/sse/official/{official_id}")
async def sse_preflight(official_id: int):
    """Handle CORS preflight request for SSE endpoint"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600"
        }
    )

# { "event": "queue_updated",  "data": { "queue_id": 5 } }
# { "event": "accepted",  "data": { "queue_id": 2 } }
 
@router.get("/sse/official/{official_id}")
async def admission_official_sse(request: Request, official_id: int):
    
    queue = asyncio.Queue()

    async def send_event(data: dict):
        await queue.put(data)

    live_chat_service.register_official_sse(official_id, send_event)

    async def event_stream():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'event': 'connected', 'message': 'SSE connection established'})}\n\n"
            
            while True:
                try:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    
                    # Wait for data with timeout to allow disconnect checking
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Send the actual event
                    event_data = {
                        "event": data.get("event", "update"),
                        "data": data
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send periodic heartbeat to keep connection alive
                    yield f"data: {json.dumps({'event': 'ping', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                    continue
                    
        except Exception as e:
            # Log error but don't expose details to client
            pass
        finally:
            live_chat_service.unregister_official_sse(official_id, send_event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Content-Type": "text/event-stream"
        }
    )


#live chat
@router.websocket("/chat/{session_id}")
async def chat_socket(websocket: WebSocket, session_id: int):
    await websocket.accept()
    await live_chat_service.join_chat(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()
            await live_chat_service.broadcast_message(
                session_id=session_id,
                sender_id=data["sender_id"],
                message=data["message"]
            )
    except WebSocketDisconnect:
        print(f"[Chat] WebSocket disconnected session={session_id}")
    finally:
        # Always clean up the WebSocket connection when it ends
        await live_chat_service.leave_chat(websocket, session_id)

@router.get("/customer/{customer_id}/sessions")
async def get_customer_sessions(customer_id: int):
    return live_chat_service.get_customer_sessions(customer_id)

@router.post("/session/rate")
async def rate_session(session_id: int, rating: int):
    """
    Customer rates a finished live chat session.
    Only session_id is required.
    """
    return await live_chat_service.rate_session(session_id, rating)
