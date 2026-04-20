from datetime import datetime
import os
from typing import Dict, List, Callable, Awaitable
from sqlalchemy.orm import Session, joinedload

from app.models.database import SessionLocal
from app.models.entities import (
    ChatSession,
    ChatInteraction,
    LiveChatQueue,
    ParticipateChatSession,
    AdmissionOfficialProfile,
    Users,
)


class LiveChatService:

    def __init__(self):
        # SSE subscribers
        self.sse_customers: Dict[int, List[Callable[[dict], Awaitable[None]]]] = {}
        self.sse_officials: Dict[int, List[Callable[[dict], Awaitable[None]]]] = {}

        # WebSocket chat connections
        self.active_sessions: Dict[int, List] = {}

    # ======================================================================
    # Helper: gửi SSE cho customer
    # ======================================================================
    async def send_customer_event(self, customer_id: int, data: dict):
        subs = self.sse_customers.get(customer_id, [])
        dead_callbacks = []
        
        for send in subs:
            try:
                await send(data)
            except Exception as e:
                print(f"Dead SSE callback for customer {customer_id}: {e}")
                dead_callbacks.append(send)
        
        # Remove dead callbacks
        for dead in dead_callbacks:
            if customer_id in self.sse_customers:
                try:
                    self.sse_customers[customer_id].remove(dead)
                except ValueError:
                    pass

    # Helper: gửi SSE cho official
    async def send_official_event(self, official_id: int, data: dict):
        subs = self.sse_officials.get(official_id, [])
        dead_callbacks = []
        
        for send in subs:
            try:
                await send(data)
            except Exception as e:
                print(f"Dead SSE callback for official {official_id}: {e}")
                dead_callbacks.append(send)
        
        # Remove dead callbacks
        for dead in dead_callbacks:
            if official_id in self.sse_officials:
                try:
                    self.sse_officials[official_id].remove(dead)
                except ValueError:
                    pass

    # Helper: đăng ký listener SSE
    def register_customer_sse(self, customer_id: int, callback):
        # Log current connections for debugging
        current_count = len(self.sse_customers.get(customer_id, []))
        print(f"Registering SSE for customer {customer_id}. Current connections: {current_count}")
        
        self.sse_customers.setdefault(customer_id, []).append(callback)
        print(f"Customer {customer_id} now has {len(self.sse_customers[customer_id])} SSE connection(s)")

    def register_official_sse(self, official_id: int, callback):
        current_count = len(self.sse_officials.get(official_id, []))
        print(f"Registering SSE for official {official_id}. Current connections: {current_count}")
        
        self.sse_officials.setdefault(official_id, []).append(callback)
        print(f"Official {official_id} now has {len(self.sse_officials[official_id])} SSE connection(s)")

    def unregister_customer_sse(self, customer_id: int, callback):
        if customer_id in self.sse_customers:
            try:
                self.sse_customers[customer_id].remove(callback)
                remaining = len(self.sse_customers[customer_id])
                print(f"Unregistered SSE for customer {customer_id}. Remaining: {remaining}")
                
                # Clean up empty lists
                if remaining == 0:
                    del self.sse_customers[customer_id]
                    print(f"Removed empty SSE list for customer {customer_id}")
            except ValueError:
                print(f"Callback not found for customer {customer_id}")

    def unregister_official_sse(self, official_id: int, callback):
        if official_id in self.sse_officials:
            try:
                self.sse_officials[official_id].remove(callback)
                remaining = len(self.sse_officials[official_id])
                print(f"Unregistered SSE for official {official_id}. Remaining: {remaining}")
                
                # Clean up empty lists
                if remaining == 0:
                    del self.sse_officials[official_id]
                    print(f"Removed empty SSE list for official {official_id}")
            except ValueError:
                print(f"Callback not found for official {official_id}")
    
    def get_sse_connection_count(self, customer_id: int = None, official_id: int = None):
        """Get SSE connection count for debugging"""
        if customer_id:
            return len(self.sse_customers.get(customer_id, []))
        if official_id:
            return len(self.sse_officials.get(official_id, []))
        return {
            "customers": {cid: len(cbs) for cid, cbs in self.sse_customers.items()},
            "officials": {oid: len(cbs) for oid, cbs in self.sse_officials.items()}
        }

    # ======================================================================
    # 1. CUSTOMER REQUEST QUEUE
    # ======================================================================
    async def customer_join_queue(self, customer_id: int):
        db = SessionLocal()
        try:
            # Check if customer is banned
            customer = db.query(Users).filter(Users.user_id == customer_id).first()
            if not customer:
                return {"error": "customer_not_found"}
            if not customer.status:
                return {"error": "customer_banned"}

            queue_entry = LiveChatQueue(
                customer_id=customer_id,
                status="waiting",
                created_at=datetime.now()
            )
            db.add(queue_entry)
            db.commit()
            db.refresh(queue_entry)

            # Gửi sự kiện cho chính student
            await self.send_customer_event(customer_id, {
                "event": "queued",
                "queue_id": queue_entry.id,
            })

            # HÀNG CHỜ CHUNG: broadcast cho TẤT CẢ official đang mở SSE
            for oid in list(self.sse_officials.keys()):
                await self.send_official_event(oid, {
                    "event": "queue_updated"
                })

            return {
                "success": True,
                "queue_id": queue_entry.id,
                "status": "waiting"
            }

        finally:
            db.close()

    # ======================================================================
    # 1B. CUSTOMER CANCEL QUEUE REQUEST
    # ======================================================================
    async def customer_cancel_queue(self, customer_id: int):
        db = SessionLocal()
        try:
            queue_entry = db.query(LiveChatQueue).filter(
                LiveChatQueue.customer_id == customer_id,
                LiveChatQueue.status == "waiting"
            ).first()
            
            if not queue_entry:
                return {"error": "no_pending_queue_request"}
            
            queue_id = queue_entry.id
            
            # Mark as canceled
            queue_entry.status = "canceled"
            db.commit()
            
            # Notify customer
            await self.send_customer_event(customer_id, {
                "event": "queue_canceled",
                "queue_id": queue_id,
                "message": "You have canceled your queue request"
            })
            
            # HÀNG CHỜ CHUNG: thông báo cho TẤT CẢ tư vấn viên
            for oid in list(self.sse_officials.keys()):
                await self.send_official_event(oid, {
                    "event": "queue_updated",
                    "message": f"Customer {customer_id} canceled their request"
                })
            
            return {"success": True, "message": "Queue request canceled successfully"}
            
        except Exception as e:
            db.rollback()
            return {"error": f"database_error: {str(e)}"}
        finally:
            db.close()

    # ======================================================================
    # 2. AO ACCEPT REQUEST
    # ======================================================================
    async def official_accept(self, official_id: int, queue_id: int):
        db = SessionLocal()
        customer_id = None
        session_id = None

        try:
            queue_item = (
            db.query(LiveChatQueue)
            .filter(LiveChatQueue.id == queue_id)
            .with_for_update()
            .first()
            )
            if not queue_item:
                return {"error": "queue_not_found"}
            if(queue_item.status != "waiting"):
                return {"error": "queue_not_available"}
            official = (
            db.query(AdmissionOfficialProfile)
            .filter_by(admission_official_id=official_id)
            .with_for_update()
            .first()
            )
            
            if not official:
                return {"error": "official_not_found"}

            if official.current_sessions >= official.max_sessions:
                return {"error": "max_sessions_reached"}

            # Store customer_id before any potential session issues
            customer_id = queue_item.customer_id
            
            print(f"[Accept] ===== ACCEPTING REQUEST =====")
            print(f"[Accept] Queue ID: {queue_id}")
            print(f"[Accept] Customer ID: {customer_id}")
            print(f"[Accept] Official ID: {official_id}")

            # Tạo live chat session
            session = ChatSession(
                session_type="live",
                start_time=datetime.now()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.chat_session_id
            
            print(f"[Accept] Created session_id: {session_id}")

            # Create participants
            participant1 = ParticipateChatSession(user_id=customer_id, session_id=session_id)
            participant2 = ParticipateChatSession(user_id=official_id, session_id=session_id)
            
            db.add_all([participant1, participant2])

            # Update queue status and official sessions
            official.current_sessions += 1
            queue_item.status = "accepted"
            db.commit()

            # Send SSE event to CUSTOMER with session_id
            print(f"[Accept] Sending 'accepted' SSE to customer {customer_id} with session_id={session_id}")
            await self.send_customer_event(customer_id, {
                "event": "accepted",
                "session_id": session_id,
                "official_id": official_id,
                "queue_id": queue_id
            })

            # SSE → update queue list for admission official
            print(f"[Accept] Sending 'queue_updated' SSE to official {official_id}")
            await self.send_official_event(official_id, {
                "event": "queue_updated"
            })
            
            # Return session_id as dict BEFORE closing db session
            result = {
                "success": True,
                "chat_session_id": session_id,
                "session_id": session_id,  # Legacy compatibility
                "customer_id": customer_id,
                "official_id": official_id,
                "queue_id": queue_id
            }
            
            db.close()
            
            print(f"[Accept] Returning result dict with session_id={session_id} to API")
            print(f"[Accept] ===== ACCEPTANCE COMPLETE =====\n")
            
            return result

        except Exception as e:
            db.rollback()
            db.close()
            print(f"ERROR in official_accept: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"internal_error: {str(e)}"}

    # ======================================================================
    # 3. AO REJECT REQUEST
    # ======================================================================
    async def official_reject(self, official_id: int, queue_id: int, reason: str):
        db = SessionLocal()
        queue_item = db.query(LiveChatQueue).filter_by(id=queue_id).first()
        if not queue_item:
            return False

        queue_item.status = "rejected"
        db.commit()

        # notify student
        await self.send_customer_event(queue_item.customer_id, {
            "event": "rejected",
            "reason": reason
        })

        # notify AO update queue
        await self.send_official_event(official_id, {
            "event": "queue_updated"
        })

        return True

    # ======================================================================
    # 4. CHAT (WebSocket)
    # ======================================================================
    async def join_chat(self, websocket, session_id: int):
        print(f"[Join Chat] New WebSocket connection for session_id={session_id}")
        print("PID:", os.getpid())
        if session_id not in self.active_sessions:
            print(f"[Join Chat] Creating new session list for session_id={session_id}")
            self.active_sessions[session_id] = []
        
        self.active_sessions[session_id].append(websocket)
        connection_count = len(self.active_sessions[session_id])
        print(f"[Join Chat] Session {session_id} now has {connection_count} active connection(s)")

        await websocket.send_json({"event": "chat_connected"})
        print(f"[Join Chat] Sent chat_connected confirmation to new connection")

    async def broadcast_message(self, session_id: int, sender_id: int, message: str):
        db = SessionLocal()

        try:
            print(f"[Broadcast] Saving message to DB: session_id={session_id}, sender_id={sender_id}, message='{message}'")
            
            chat = ChatInteraction(
                session_id=session_id,
                sender_id=sender_id,
                message_text=message,
                timestamp=datetime.now().date(),  # Use .date() for Date column
                is_from_bot=False
            )
            db.add(chat)
            db.commit()
            db.refresh(chat)
            
            print(f"[Broadcast] Message saved with interaction_id={chat.interaction_id}")

            payload = {
                "event": "message",
                "session_id": session_id,
                "sender_id": sender_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "interaction_id": chat.interaction_id
            }

            # Send to all connections in this session
            active_connections = self.active_sessions.get(session_id, [])
            print(f"[Broadcast] Active connections for session {session_id}: {len(active_connections)}")
            
            if len(active_connections) == 0:
                print(f"[Broadcast] WARNING: No active WebSocket connections for session {session_id}!")
            
            for idx, conn in enumerate(active_connections):
                try:
                    print(f"[Broadcast] Sending to connection #{idx+1}...")
                    await conn.send_json(payload)
                    print(f"[Broadcast] Successfully sent to connection #{idx+1}")
                except Exception as e:
                    print(f"[Broadcast] Error sending to connection #{idx+1}: {e}")
                    # Remove broken connections
                    if conn in self.active_sessions[session_id]:
                        self.active_sessions[session_id].remove(conn)
                        print(f"[Broadcast] Removed broken connection #{idx+1}")

        except Exception as e:
            db.rollback()
            print(f"Error saving/broadcasting message: {e}")
            raise  # Re-raise to let WebSocket handler know
        finally:
            db.close()

    async def leave_chat(self, websocket, session_id: int):
        """Remove WebSocket connection from active session"""
        print(f"[Leave Chat] Removing connection from session_id={session_id}")
        
        if session_id in self.active_sessions:
            if websocket in self.active_sessions[session_id]:
                self.active_sessions[session_id].remove(websocket)
                remaining = len(self.active_sessions[session_id])
                print(f"[Leave Chat] Connection removed. Remaining connections: {remaining}")
            else:
                print(f"[Leave Chat] WARNING: WebSocket not found in session {session_id}")
            
            # Clean up empty session lists
            if not self.active_sessions[session_id]:
                del self.active_sessions[session_id]
                print(f"[Leave Chat] Session {session_id} has no more connections, removed from active_sessions")
        else:
            print(f"[Leave Chat] WARNING: Session {session_id} not found in active_sessions")


    # ===============================================================
    # 5. END SESSION
    # ===============================================================
    async def end_session(self, session_id: int, ended_by: int):
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter_by(chat_session_id=session_id).first()
            if not session:
                return {"error": "session_not_found"}
            
            # Đã kết thúc rồi thì thôi
            if session.end_time is not None:
                return {"error": "session_already_ended"}

            # Người kết thúc phải là participant
            participant = db.query(ParticipateChatSession).filter(
                ParticipateChatSession.session_id == session_id,
                ParticipateChatSession.user_id == ended_by
            ).first()
            if not participant:
                return {"error": "not_session_participant"}

            # Lấy toàn bộ participant để:
            # - tìm official
            # - gửi SSE cho tất cả
            all_participants = db.query(ParticipateChatSession).filter(
                ParticipateChatSession.session_id == session_id
            ).all()

            # Tìm official (nếu có)
            official_id = None
            for p in all_participants:
                profile = db.query(AdmissionOfficialProfile).filter_by(
                    admission_official_id=p.user_id
                ).first()
                if profile:
                    official_id = p.user_id
                    break

            # Kết thúc phiên
            session.end_time = datetime.now().date()

            # Giảm current_sessions của official
            if official_id:
                profile = db.query(AdmissionOfficialProfile).filter_by(
                    admission_official_id=official_id
                ).first()
                if profile and profile.current_sessions > 0:
                    profile.current_sessions -= 1

            db.commit()

            # Payload chung thông báo kết thúc
            payload = {
                "event": "chat_ended",
                "session_id": session_id,
                "ended_by": ended_by
            }

            # 1️⃣ Gửi qua WebSocket cho tất cả connection trong session
            for conn in self.active_sessions.get(session_id, []):
                try:
                    await conn.send_json(payload)
                except Exception as e:
                    print(f"[End Session] WS send error: {e}")

            # 2️⃣ Gửi qua SSE cho tất cả user tham gia (học sinh + officer nếu đang mở SSE)
            participant_ids = [p.user_id for p in all_participants]
            for uid in participant_ids:
                try:
                    await self.send_customer_event(uid, payload)
                    await self.send_official_event(official_id, payload)
                except Exception as e:
                    print(f"[End Session] SSE error for user {uid}: {e}")

            # # Dọn WebSocket
            # self.active_sessions.pop(session_id, None)

            return {"success": True}

        except Exception as e:
            db.rollback()
            return {"error": f"database_error: {str(e)}"}
        finally:
            db.close()

    def get_my_status(self, customer_id: int):
        db = SessionLocal()
        item = db.query(LiveChatQueue) \
            .filter_by(customer_id=customer_id) \
            .order_by(LiveChatQueue.created_at.desc()) \
            .first()
        db.close()
        return item

    def delete_queue_item(self, queue_id: int):
        db = SessionLocal()
        item = db.query(LiveChatQueue).filter_by(id=queue_id).first()
        if item:
            db.delete(item)
            db.commit()
        db.close()
        return True

    def get_queue_list(self, official_id: int):
        """Hàng chờ chung cho tất cả tư vấn viên"""
        db = SessionLocal()
        items = db.query(LiveChatQueue).filter(
            LiveChatQueue.status == "waiting"
        ).options(
            joinedload(LiveChatQueue.customer)
        ).all()
        official = (
            db.query(AdmissionOfficialProfile)
            .filter_by(admission_official_id=official_id)
            .with_for_update()
            .first()
            )
            
        result = []
        if not official:
            return result
        for item in items:
            queue_item_dict = {
                'id': item.id,
                'customer_id': item.customer_id,
                'status': item.status,
                'created_at': item.created_at.isoformat() if item.created_at else None,
                'customer': {
                    'full_name': item.customer.full_name if item.customer else f'Customer {item.customer_id}',
                    'email': item.customer.email if item.customer else 'N/A',
                    'phone_number': item.customer.phone_number if item.customer else 'N/A'
                } if item.customer else {
                    'full_name': f'Customer {item.customer_id}',
                    'email': 'N/A',
                    'phone_number': 'N/A'
                }
            }
            result.append(queue_item_dict)
        
        db.close()
        return result

    async def get_active_sessions(self, official_id: int):
        """Get all active chat sessions for an admission official"""
        db = SessionLocal()
        try:
            active_sessions_query = db.query(
                ChatSession.chat_session_id,
                ChatSession.start_time,
                ChatSession.session_type
            ).join(
                ParticipateChatSession, 
                ChatSession.chat_session_id == ParticipateChatSession.session_id
            ).filter(
                ParticipateChatSession.user_id == official_id,
                ChatSession.start_time.isnot(None),
                ChatSession.end_time.is_(None)
            ).all()
            
            result = []
            for session in active_sessions_query:
                session_id, start_time, session_type = session
                
                customer_participant = db.query(
                    ParticipateChatSession, Users.full_name
                ).join(
                    Users, ParticipateChatSession.user_id == Users.user_id
                ).filter(
                    ParticipateChatSession.session_id == session_id,
                    ParticipateChatSession.user_id != official_id
                ).first()
                
                if customer_participant:
                    participant, customer_name = customer_participant
                    
                    result.append({
                        'session_id': session_id,
                        'customer_id': participant.user_id,
                        'customer_name': customer_name,
                        'session_type': session_type or 'live',
                        'start_time': start_time.isoformat() + 'T00:00:00' if start_time else datetime.now().isoformat(),
                        'status': 'active'
                    })
            
            return result
            
        finally:
            db.close()
    
    def get_messages(self, session_id: int):
        db = SessionLocal()
        msgs = db.query(ChatInteraction) \
            .filter_by(session_id=session_id) \
            .order_by(ChatInteraction.timestamp.asc()) \
            .all()
        db.close()
        return msgs
    
    def get_customer_sessions(self, customer_id: int):
        db = SessionLocal()
        try:
            sessions = (
                db.query(ChatSession)
                .join(
                    ParticipateChatSession,
                    ChatSession.chat_session_id == ParticipateChatSession.session_id,
                )
                .filter(
                    ParticipateChatSession.user_id == customer_id,
                    ChatSession.session_type == "live",
                )
                .order_by(ChatSession.start_time.desc())
                .all()
            )

            result = []
            for s in sessions:
                result.append({
                    "session_id": s.chat_session_id,
                    "start_time": s.start_time.isoformat() if s.start_time else None,
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "status": "active" if s.end_time is None else "ended",
                })
            return result
        finally:
            db.close()

    async def rate_session(self, session_id: int, rating: int):
        db = SessionLocal()
        try:
            # 1. Validate rating
            if rating < 1 or rating > 5:
                return {"error": "invalid_rating"}

            # 2. Check session
            session = db.query(ChatSession).filter_by(
                chat_session_id=session_id
            ).first()

            if not session:
                return {"error": "session_not_found"}

            if session.end_time is None:
                return {"error": "session_not_ended"}

            if session.feedback_rating is not None:
                return {"error": "already_rated"}

            # 3. Find official via session participants
            official_participant = (
                db.query(ParticipateChatSession)
                .join(
                    AdmissionOfficialProfile,
                    AdmissionOfficialProfile.admission_official_id
                    == ParticipateChatSession.user_id
                )
                .filter(ParticipateChatSession.session_id == session_id)
                .first()
            )

            if not official_participant:
                return {"error": "official_not_found"}

            official_id = official_participant.user_id

            # 4. Save rating to session
            session.feedback_rating = rating
            db.flush()  # <-- FIX 1

            # 5. Recalculate official average rating
            ratings = (
                db.query(ChatSession.feedback_rating)
                .join(
                    ParticipateChatSession,
                    ChatSession.chat_session_id == ParticipateChatSession.session_id
                )
                .filter(
                    ParticipateChatSession.user_id == official_id,
                    ChatSession.feedback_rating.isnot(None)
                )
                .all()
            )

            rating_values = [r[0] for r in ratings]
            print("DEBUG ratings:", ratings)
            print("DEBUG rating_values:", rating_values)
            
            if not rating_values:   # <-- FIX 2
                new_avg = float(rating)
            else:
                new_avg = round(sum(rating_values) / len(rating_values), 1)

            official_profile = db.query(AdmissionOfficialProfile).filter_by(
                admission_official_id=official_id
            ).first()

            official_profile.rating = new_avg

            db.commit()


            return {
                "success": True,
                "session_id": session_id,
                "official_id": official_id,
                "rating": rating,
                "official_avg_rating": new_avg
            }

        except Exception as e:
            db.rollback()
            import traceback
            traceback.print_exc()
            return {"error": f"database_error: {str(e)}"}
        finally:
            db.close()
