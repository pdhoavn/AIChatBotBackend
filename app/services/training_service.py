from datetime import datetime
from typing import Any, Dict, List, Optional
import time
import json
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import asyncio
from sqlalchemy.orm import Session
from app.models import schemas
from app.models.entities import AcademicScore, ChatInteraction, ChatSession, DocumentChunk, FaqStatistics, Intent, KnowledgeBaseDocument, Major, ParticipateChatSession, RiasecResult, TargetAudience, TrainingQuestionAnswer
from app.models.database import SessionLocal
from sqlalchemy.exc import SQLAlchemyError
from app.services.memory_service import MemoryManager
from app.utils.document_processor import DocumentProcessor

memory_service = MemoryManager()

class TrainingService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=self.openai_api_key,
            temperature=0.7
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=self.openai_api_key
        )
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            timeout=30,
            check_compatibility=False,
        )
        self.training_qa_collection = "training_qa"
        self.documents_collection = "knowledge_base_documents"
        self.university_name = os.getenv("CHAT_UNIVERSITY_NAME", "Đại học Giao thông Vận tải")
        self._init_collections()

    def _init_collections(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.training_qa_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
        except:
            pass

        try:
            self.qdrant_client.create_collection(
                collection_name=self.documents_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
        except:
            pass

    def _debug_log(self, message: str, trace_id: Optional[str] = None):
        if trace_id:
            print(f"[RAG][{trace_id}] {message}")
            return
        print(f"[RAG] {message}")

    def create_chat_session(self, user_id: int, session_type: str = "chatbot") -> int:
        """
        Tạo chat session mới
        
        Args:
            user_id: ID của user
            session_type: "chatbot" hoặc "live"
        
        Returns:
            session_id: ID của session vừa tạo
        """
        
        db = SessionLocal()
        if not user_id:
            session = ChatSession(
                session_type=session_type,
                start_time=datetime.now()
            )
            db.add(session)
            db.flush()
            db.commit()
            return session.chat_session_id
        try:
            session = ChatSession(
                session_type=session_type,
                start_time=datetime.now()
            )
            db.add(session)
            db.flush()
            
            # Add user vào participate table
            participate = ParticipateChatSession(
                user_id=user_id,
                session_id=session.chat_session_id
            )
            db.add(participate)
            db.commit()
            
            return session.chat_session_id
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error creating session: {e}")
            raise
        finally:
            db.close()

    def get_session_history(self, session_id: int, limit: int = 50) -> List[Dict]:
        """
        Lấy lịch sử chat của session
        
        Returns:
            List of messages [{message_text, timestamp, is_from_bot}, ...]
        """
        db = SessionLocal()
        try:
            interactions = db.query(ChatInteraction).filter(
                ChatInteraction.session_id == session_id
            ).order_by(
                ChatInteraction.timestamp.asc()
            ).limit(limit).all()
            
            return [
                {
                    "message_text": i.message_text,
                    "timestamp": i.timestamp.isoformat() if i.timestamp else None,
                    "is_from_bot": i.is_from_bot,
                    "rating": i.rating
                }
                for i in interactions
            ]
        finally:
            db.close()
    
    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """
        Lấy tất cả sessions của user (để hiển thị recent chats)
        
        Returns:
            List of sessions với preview message cuối cùng
        """
        db = SessionLocal()
        try:
            sessions = db.query(ChatSession).filter(ChatSession.session_type == "chatbot").join(
                ParticipateChatSession
            ).filter(
                ParticipateChatSession.user_id == user_id
            ).order_by(
                ChatSession.start_time.desc()
            ).all()
            
            result = []
            for session in sessions:
                # Lấy message cuối cùng làm preview
                last_msg = db.query(ChatInteraction).filter(
                    ChatInteraction.session_id == session.chat_session_id
                ).order_by(
                    ChatInteraction.timestamp.desc()
                ).first()
                
                result.append({
                    "session_id": session.chat_session_id,
                    "session_type": session.session_type,
                    "start_time": session.start_time.isoformat() if session.start_time else None,
                    "last_message_preview": last_msg.message_text[:50] + "..." if last_msg else "",
                    "last_message_time": last_msg.timestamp.isoformat() if last_msg and last_msg.timestamp else None
                })
            
            return result
        finally:
            db.close()

    def delete_chat_session(self, session_id: int, user_id: Optional[int] = None) -> bool:
        """
        Xóa 1 session chat:
        - Nếu có user_id: chỉ xóa session thuộc về user đó
        - Nếu không có user_id: xóa theo session_id (guest session)

        Trả về:
            True  nếu xóa được
            False nếu không tìm thấy session
        """
        db = SessionLocal()
        try:
            query = db.query(ChatSession)

            # Nếu có user_id thì check session thuộc user đó
            if user_id:
                query = query.join(ParticipateChatSession).filter(
                    ParticipateChatSession.user_id == user_id
                )

            session = query.filter(
                ChatSession.chat_session_id == session_id
            ).first()

            if not session:
                return False

            # Do ChatSession định nghĩa cascade="all, delete-orphan"
            # nên xóa session sẽ tự xóa ChatInteraction & ParticipateChatSession liên quan
            db.delete(session)
            db.commit()
            return True

        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error deleting session: {e}")
            raise
        finally:
            db.close()

    # ---------------------------
    # Query enrichment: dùng chat_history + last bot question để build a full query
    # ---------------------------
    async def enrich_query(self, session_id: str, user_message: str) -> str:
        memory = memory_service.get_memory(session_id)
        mem_vars = memory.load_memory_variables({})
        chat_history = mem_vars.get("chat_history", "")

        prompt = f"""
        Bạn là một trợ lý chuẩn hóa truy vấn cho chatbot RAG tư vấn tuyển sinh {self.university_name}.

        Cuộc hội thoại gần đây:
        {chat_history}

        Phản hồi mới nhất của người dùng:
        "{user_message}"

        NHIỆM VỤ:
        - Chỉ viết lại câu hỏi của người dùng cho rõ ràng hơn nếu:
        • Câu trả lời hiện tại phụ thuộc trực tiếp vào hội thoại trước đó
        • Hoặc người dùng dùng đại từ, câu rút gọn, câu thiếu chủ ngữ

        - TUYỆT ĐỐI KHÔNG:
        • Thêm thông tin mới
        • Thêm phạm vi mới (ví dụ: “các trường khác”, “tại Việt Nam”, “so sánh”)
        • Thay đổi mục tiêu câu hỏi
        • Suy đoán ý định người dùng

        - Nếu câu hỏi đã rõ ràng và độc lập:
        → Trả về NGUYÊN VĂN phản hồi mới nhất của người dùng.

        - Chỉ xuất ra MỘT câu truy vấn tiếng Việt, không giải thích.

        """
        # assume async predict exists
        enriched = await self.llm.ainvoke(prompt)
        print("==== RAW RESPONSE ====")
        print(enriched.content)
        print("======================")
        # fallback: if empty use original
        enriched_txt = (enriched.content or "").strip().splitlines()[0] if enriched else user_message
        return enriched_txt   

    # ---------------------------
    # LLM relevance check: ensure enriched_query actually matches the training QA
    # ---------------------------
    async def llm_relevance_check(self, enriched_query: str, matched_question: str, answer: str) -> bool:
        prompt = f"""
        Bạn là chuyên gia đánh giá giữa câu hỏi tìm kiếm, câu hỏi trong cơ sở dữ liệu và câu trả lời cho 1 hệ thống chat RAG tuyển sinh, hãy suy luận. 

        Câu hỏi tìm kiếm (đã chuẩn hóa): "{enriched_query}"
        Câu hỏi DB: "{matched_question}"
        Câu trả lời chính thức: "{answer}"

        Hãy trả lời duy nhất chỉ một từ: "true" nếu câu hỏi DB phù hợp và trả lời đó hợp lý cho truy vấn tìm kiếm; "false" nếu chỉ trùng từ khóa hoặc không phù hợp.
        Hoặc có thể trả về "true" nếu câu hỏi tìm kiếm chỉ là lời chào.
        """
        res = await self.llm.ainvoke(prompt)
        if not res.content:
            return False
        r = res.content.strip().lower()
        return ("đúng" in r) or ("true" in r) or (r.startswith("đúng")) or (r.startswith("true"))

    async def llm_document_recommendation_check(self, enriched_query: str, context: str) -> bool:
        prompt = f"""
        Bạn là hệ thống kiểm tra 2 tầng:
        - Tầng 1 là hệ thống kiểm tra mức độ liên quan giữa câu hỏi người dùng và nội dung trong Document Base (RAG) cho chatbot RAG tư vấn tuyển sinh.
        - Tầng 2 là hệ thống kiểm tra mức độ liên quan giữa câu hỏi người dùng có liên quan đến các nội dung tư vấn ngành học hay tư vấn cho cá nhân dựa theo hồ sơ của học sinh hoặc những câu liên quan đến RIASEC, học bạ, GPA, sở thích, nguyện vọng cá nhân; hoặc yêu cầu so sánh ngành theo profile; hoặc yêu cầu gợi ý ngành phù hợp cho chatbot RAG tư vấn tuyển sinh.
        Yêu cầu kiểm tra câu hỏi người dùng có phù hợp với tầng 1 hoặc tầng 2:
        - Nếu phù hợp với tầng 1 thì trả về duy nhất 1 từ "document"
        - Nếu phù hợp với tầng 2 thì trả về duy nhất 1 từ "recommendation"
        - Nếu tầng 1 phù hợp thì không cần check đến tầng 2
        - Nếu không phù hợp với tầng 1 và tầng 2 thì trả về duy nhất 1 từ "Nope"
        - Check tầng 1(document) đầu tiên:
        - Chỉ trả về "document" nếu:
          • Document base có dữ liệu TRỰC TIẾP để trả lời
          • Không suy diễn
          • Không trả lời chung chung
          • Không chỉ dựa vào trùng từ khóa
        - Chỉ trả về "document" nếu NỘI DUNG của document base THỰC SỰ có thông tin trả lời câu hỏi và thông tin đó đúng ý định của người dùng muốn biết
        - Trước khi trả về "document", hãy tự hỏi và suy luận kĩ càng:
              "Nội dung của Document base có trực tiếp liệt kê hoặc mô tả thông tin mà người dùng hỏi không?"
        - Check qua tầng 2 nếu:
            • chỉ trùng từ khóa nhưng không cùng ý nghĩa
            • document không chứa dữ liệu cần thiết để trả lời
            • truy vấn là yêu cầu tư vấn cá nhân (Recommendation), không phải tìm kiến thức
            • query chung chung như: "tôi hợp ngành nào", "hãy tư vấn", "mô tả về tôi", "nên học gì"
            • context không cung cấp thông tin trực tiếp liên quan
        - Check tầng 2(recommendation):
        - Chỉ trả về "recommendation" nếu câu hỏi người dùng liên quan đến các nội dung tư vấn ngành học hay tư vấn cho cá nhân dựa theo hồ sơ của học sinh hoặc những câu liên quan đến RIASEC, học bạ, GPA, sở thích, năng lực, trình độ, học lực, nguyện vọng cá nhân; hoặc yêu cầu so sánh ngành theo profile; hoặc yêu cầu gợi ý ngành phù hợp. Và các câu hỏi đề cập đến năng lực tự đánh giá của người học (ví dụ: học lực yếu, trung bình, kém môn nào, không có năng khiếu, sợ không theo kịp, lo lắng về khả năng học) thì vẫn được coi là tư vấn cá nhân, kể cả khi không có GPA hoặc hồ sơ chi tiết. Nếu câu hỏi thể hiện nhu cầu được tư vấn định hướng cho cá nhân nhưng thiếu thông tin chi tiết, vẫn trả về "recommendation" để chatbot hỏi thêm thông tin.
        - Chỉ trả về "Nope" khi cả tầng 1 và tầng 2 đều không liên quan đến câu hỏi người dùng.
        
        Câu hỏi người dùng: "{enriched_query}"

        Nội dung Document Base (context):
        \"\"\"
        {context}
        \"\"\"

        
        """

        res = await self.llm.ainvoke(prompt)
        r = res.content.strip().lower()
        if r not in ["document", "recommendation", "nope"]:
            r = "nope"
        return r

    async def llm_suitable_for_recommedation_check(self, enriched_query: str, context: str) -> bool:
        prompt = f"""
        Bạn là hệ thống kiểm tra mức độ liên quan giữa câu hỏi người dùng có liên quan đến các nội dung tư vấn ngành học hay tư vấn cho cá nhân dựa theo hồ sơ của học sinh hoặc những câu liên quan đến RIASEC, học bạ, GPA, sở thích, nguyện vọng cá nhân; hoặc yêu cầu so sánh ngành theo profile; hoặc yêu cầu gợi ý ngành phù hợp cho chatbot RAG tư vấn tuyển sinh.

        Yêu cầu:
        - Chỉ trả về "true" nếu câu hỏi có liên quan đến các nội dung đó.
        - Trả về "false" nếu câu hỏi không liên quan đến các nội dung đó.

        Câu hỏi người dùng: "{enriched_query}"

        
        Hãy TRẢ LỜI DUY NHẤT:
        - "true" → nếu câu hỏi có liên quan đến các nội dung đó 
        - "false" → nếu câu hỏi không liên quan đến các nội dung đó
        """

        res = await self.llm.ainvoke(prompt)
        if not res.content:
            return False
        r = res.content.strip().lower()
        return ("đúng" in r) or ("true" in r) or (r.startswith("đúng")) or (r.startswith("true"))

    async def response_from_riasec_result(self, riasec_result: schemas.RiasecResultCreate):
        prompt = f"""
        Bạn là chuyên gia hướng nghiệp Holland (RIASEC).

        Dưới đây là điểm RIASEC của người dùng:
        - Realistic (R): {riasec_result.score_realistic}
        - Investigative (I): {riasec_result.score_investigative}
        - Artistic (A): {riasec_result.score_artistic}
        - Social (S): {riasec_result.score_social}
        - Enterprising (E): {riasec_result.score_enterprising}
        - Conventional (C): {riasec_result.score_conventional}

        Yêu cầu:
        1. Tự xác định mã RIASEC chính của người dùng bằng cách chọn 3 nhóm có điểm cao nhất (ví dụ: “ISA”, “REI”, “SEC”…).
        2. Giải thích ý nghĩa mã RIASEC đó theo phong cách hướng nghiệp.
        3. Tóm tắt đặc điểm tính cách chính (3–5 câu).
        4. Trả lời bằng tiếng Việt, sử dụng Markdown (tiêu đề, gạch đầu dòng, xuống dòng rõ ràng).

        Trả về:
        - Một đoạn văn hoàn chỉnh, bao gồm cả mã RIASEC mà bạn suy luận.
            """

        try:
            res = await self.llm.ainvoke(prompt)
            return res.content.strip()

        except Exception as e:
            print("LLM error:", e)
            return "Xin lỗi, hệ thống tạm thời chưa thể phân tích kết quả RIASEC. Bạn vui lòng thử lại sau."

    async def load_session_history_to_memory(self, session_id: int, db: Session):
        memory = memory_service.get_memory(session_id)

        # Lấy lịch sử chat theo thứ tự thời gian
        interactions = (
            db.query(ChatInteraction)
            .filter(ChatInteraction.session_id == session_id)
            .order_by(ChatInteraction.timestamp.asc())
            .all()
        )

        last_user_msg = None
        for inter in interactions:
            if not inter.is_from_bot:
                # user message
                last_user_msg = inter.message_text
            else:
                # bot message -> kết hợp với user message trước đó (nếu có)
                memory.save_context(
                    {"input": last_user_msg or ""},
                    {"output": inter.message_text}
                )
                last_user_msg = None

        # Nếu cuối cùng là tin nhắn user chưa được phản hồi
        if last_user_msg:
            memory.save_context({"input": last_user_msg}, {"output": ""})

    def update_faq_statistics(self, db: Session, response_id: int, intent_id: int = 1):
        
        try:
            response = db.query(ChatInteraction).filter(
            ChatInteraction.interaction_id == response_id,
            ChatInteraction.is_from_bot == True
        ).first()

            if not response:
                raise ValueError("Chatbot response not found")

            faq = FaqStatistics(
                response_from_chat_id = response_id,
                intent_id = intent_id
            )
            db.add(faq)
            db.commit()

        except Exception as e:
            db.rollback()
            print(f"Error updating FaqStatistics: {e}")
            

    async def stream_response_from_context(self, query: str, context: str, session_id: int, user_id: int, intent_id: int, message: str):
        db = SessionLocal()
        
        try:
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            else:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=user_id,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")
            

            prompt = f"""Bạn là một tư vấn viên tuyển sinh chuyên nghiệp của trường {self.university_name}
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === THÔNG TIN THAM KHẢO ===
            {context}
            === CÂU HỎI ===
            {query}
            === HƯỚNG DẪN ===
            - Trả lời bằng tiếng Việt
            - Dựa vào thông tin tham khảo trên được cung cấp
            - Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
            - Trả lời theo định dạng Markdown: dùng tiêu đề ##, gạch đầu dòng -, xuống dòng rõ ràng.
            - Hãy tạo ra câu trả lời không quá dài, gói gọn ý chính, chỉ khi câu hỏi yêu cầu "chi tiết" thì mới tạo câu trả lời đầy đủ
            - Bạn là tư vấn tuyển sinh của trường {self.university_name}, nếu câu hỏi yêu cầu thông tin của một trường khác thì nói rõ là không có dữ liệu trong hệ thống hiện tại
            - Nếu không tìm thấy thông tin, hãy nói rõ và gợi ý liên hệ trực tiếp nhân viên tư vấn
            - Không cần phải chào hỏi mỗi lần trả lời, vào thẳng vấn đề chính
            - Nếu câu hỏi chỉ là chào hỏi, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
            - Khi có thể, hãy **giải thích thêm bối cảnh hoặc gợi ý bước tiếp theo**, ví dụ:  
                “Bạn muốn mình gửi danh sách ngành đào tạo kèm chuyên ngành chi tiết không?”  
                hoặc  
                “Nếu bạn quan tâm học bổng, mình có thể nói rõ các loại học bổng hiện có nhé!”
            """
            full_response = ""
            async for chunk in self.llm.astream(prompt):
                text = chunk.content or ""
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop
            print(full_response)
            memory.save_context({"input": query}, {"output": full_response})  
            
            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            self.update_faq_statistics(db, bot_msg.interaction_id, intent_id = intent_id)
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close()

    async def stream_response_from_qa(self, query: str, context: str, session_id: int = 1, user_id: int = 1, intent_id: int = 1, message: str = ""):
        db = SessionLocal()
        try:
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            else:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=user_id,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")

            prompt = f"""
            Bạn là chatbot tư vấn tuyển sinh của trường {self.university_name}.
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === CÂU TRẢ LỜI CHÍNH THỨC ===
            {context}

            === CÂU HỎI NGƯỜI DÙNG ===
            {query}

            === HƯỚNG DẪN TRẢ LỜI ===
            - Trả lời theo định dạng Markdown: dùng tiêu đề ##, gạch đầu dòng -, xuống dòng rõ ràng.
            - Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
            - Hãy trả lời chính xác bằng "CÂU TRẢ LỜI CHÍNH THỨC" mà KHÔNG SUY DIỄN THÊM.
            - Bạn là tư vấn tuyển sinh của trường {self.university_name}, nhớ kiểm tra kĩ rõ ràng câu hỏi, nếu câu hỏi yêu cầu thông tin của một trường khác thì nói rõ là không có dữ liệu trong hệ thống hiện tại
            - Nếu câu hỏi chỉ là chào hỏi, hỏi thời tiết, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
            - Không cần phải chào hỏi mỗi lần trả lời, vào thẳng vấn đề chính
            """
            full_response = ""
            async for chunk in self.llm.astream(prompt):
                text = chunk.content or ""
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop

            memory.save_context({"input": query}, {"output": full_response})  
            print("Saved to memory. Current messages:", len(memory.chat_memory.messages))

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            
            self.update_faq_statistics(db, bot_msg.interaction_id, intent_id = intent_id)
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close() 
    
    async def stream_response_from_recommendation(
        self,
        user_id: int,
        session_id: int,
        query: str,
        message: str
    ):
        db = SessionLocal()
        try:
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            else:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=user_id,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")

            user_profile = self._get_user_personality_and_academics(user_id, db)
            majors = self._get_all_majors_and_specialization_from_db(db, limit=200)

            personality = user_profile.get("personality_summary") or ""
            academic_summary = user_profile.get("academic_summary") or ""
            gpa = user_profile.get("gpa", "")

            maj_texts = []
            for m in majors:
                line = f"- [{m['major_id']}]: {m['major_name']}"
                
                if m["specializations"]:
                    for s in m["specializations"]:
                        line += f"\n    • {s['specialization_name']}"
                
                maj_texts.append(line)

            prompt = f"""
        Bạn là chatbot tư vấn tuyển sinh của trường {self.university_name}. Nhiệm vụ của bạn là tư vấn chọn ngành:
        **CHỈ tư vấn chọn ngành khi câu hỏi của người dùng thật sự liên quan.**
        
        Đây là đoạn hội thoại trước: 
            {chat_history}
        ===========================
        ### THÔNG TIN HỒ SƠ NGƯỜI DÙNG
        Personality summary(RIASEC Result):
        {personality}

        Academic summary(học bạ):
        {academic_summary}

        

        ===========================
        ### DANH SÁCH CÁC NGÀNH
        {chr(10).join(maj_texts)}

        ===========================
        ### CÂU HỎI NGƯỜI DÙNG
        "{query}"

        ===========================
        ### HƯỚNG DẪN XỬ LÝ

        1. **Đầu tiên, hãy kiểm tra xem câu hỏi có thật sự liên quan đến việc tư vấn chọn ngành hay không, hoặc câu hỏi có liên quan đến thông tin hồ sơ người dùng hay không, hoặc câu hỏi có liên quan năng lực của người hỏi hay không hoặc các câu hỏi đề cập đến năng lực tự đánh giá của người học (ví dụ: học lực yếu, trung bình, kém môn nào, không có năng khiếu, sợ không theo kịp, lo lắng về khả năng học) thì vẫn được coi là tư vấn cá nhân, kể cả khi không có GPA hoặc hồ sơ chi tiết.**
        - Nếu KHÔNG liên quan → bạn hãy tự tạo câu phản hồi phù hợp với CÂU HỎI NGƯỜI DÙNG
        2. Nếu câu hỏi có liên quan đến thông tin hồ sơ người dùng ở trên bao gồm RIASEC Result và học bạ mà hồ sơ người dùng trống thì hãy yêu cầu người dùng nhập những thông tin này như RIASEC Result hoặc học bạ, 1 trong 2 là có thể được tư vấn dựa vào thông tin hồ sơ người dùng. Đề xuất theo tính cách có thể dựa vào kết quả RIASEC Result của THÔNG TIN HỒ SƠ NGƯỜI DÙNG
        3. Trả lời theo định dạng Markdown: dùng tiêu đề ##, gạch đầu dòng -, xuống dòng rõ ràng.
        4. Nếu câu hỏi không liên quan thì hãy từ chối yêu cầu và đề nghị nhắn trực tiếp bên tuyển sinh
        5. Không cần phải chào hỏi mỗi lần trả lời, vào thẳng vấn đề chính
        6. Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
        """
            full_response = ""
            async for chunk in self.llm.astream(prompt):
                text = chunk.content or ""
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop

            memory.save_context({"input": query}, {"output": full_response})  
            print("Saved to memory. Current messages:", len(memory.chat_memory.messages))

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close()

    async def stream_response_from_NA(self, query: str, context: str, session_id: int = 1, user_id: int = 1, intent_id: int = 0, message: str = "", current_audience_id: int = None,
current_intent_id: int = None):
        db = SessionLocal()
        try:
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            else:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=user_id,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")
            suggestion = self.cross_scope_search(query)
            print("Suggestion raw:", suggestion)
            if suggestion:
                if (
                    suggestion["audience_ids"] == current_audience_id and
                    suggestion["intent_id"] == current_intent_id
                ):
                    suggestion = None

            # ===== BUILD RESPONSE (KHÔNG DÙNG LLM nếu có suggestion) =====
            if suggestion:
                if suggestion["source"] == "training_qa":
                    
                    response_text = f"""
                    ## Không tìm thấy thông tin trong mục hiện tại.


                    Mình đã kiểm tra trong phạm vi **đối tượng** và **lĩnh vực** bạn đang chọn, 
                    nhưng hiện tại hệ thống chưa có dữ liệu phù hợp để trả lời chính xác câu hỏi này.


                    Gợi ý nội dung liên quan:                 


                    Mình phát hiện câu hỏi của bạn có thể thuộc phạm vi khác trong hệ thống: 


                    - **Đối tượng phù hợp**: {suggestion['audience_names']}

                    - **Lĩnh vực liên quan**: {suggestion['intent_name']}

                     .Bạn có thể làm gì tiếp theo?

                    - **Chuyển sang đúng đối tượng / lĩnh vực** để xem thông tin chính xác hơn

                    - Tiếp tục đặt câu hỏi chi tiết hơn (ví dụ: nội dung cụ thể bạn muốn biết)

                    - Nếu cần hỗ trợ sâu hơn, bạn có thể liên hệ trực tiếp bộ phận tư vấn của trường

                    """
                else:
                    preview = suggestion.get("chunk_preview", "")[:200]

                    response_text = f"""
                    ## Không tìm thấy thông tin trong mục hiện tại.
                    Mình đã kiểm tra trong phạm vi **đối tượng** và **lĩnh vực** bạn đang chọn, 
                    nhưng hiện tại hệ thống chưa có dữ liệu phù hợp để trả lời chính xác câu hỏi này.
                    Gợi ý nội dung liên quan \n
                   
                    Mình phát hiện câu hỏi của bạn có thể thuộc phạm vi khác trong hệ thống:
                    - **Đối tượng phù hợp**: {suggestion['audience_names']}
                    - **Lĩnh vực liên quan**: {suggestion['intent_name']}

                    📄 Nội dung gần đúng:
                    "{preview}..."

                    Bạn có thể làm gì tiếp theo?

                    - **Chuyển sang đúng đối tượng / lĩnh vực** để xem thông tin chính xác hơn
                    - Tiếp tục đặt câu hỏi chi tiết hơn (ví dụ: nội dung cụ thể bạn muốn biết)
                    - Nếu cần hỗ trợ sâu hơn, bạn có thể liên hệ trực tiếp bộ phận tư vấn của trường

                    """
                    # 👉 stream giả lập (không cần LLM)
                for token in response_text.split():
                    yield token + " "
                    await asyncio.sleep(0)

                full_response = response_text

            else:


                prompt = f"""
                Bạn là chatbot tư vấn tuyển sinh của trường {self.university_name}.
                Đây là đoạn hội thoại trước: 
                {chat_history}
                === CÂU TRẢ LỜI CHÍNH THỨC ===
                {context}

                === CÂU HỎI NGƯỜI DÙNG ===
                {query}

                === HƯỚNG DẪN TRẢ LỜI ===
                Bạn là tầng phản hồi của chatbot tư vấn tuyển sinh {self.university_name}.

                Nhiệm vụ của bạn KHÔNG phải trả lời kiến thức,
                mà là xử lý tình huống, tự tạo câu phản hồi phù hợp với CÂU HỎI NGƯỜI DÙNG khi NGỮ CẢNH ĐƯỢC CUNG CẤP
                KHÔNG PHÙ HỢP hoặc CHƯA CÓ DATA với ý định câu hỏi người dùng.
                ## Hướng xử lý
                - Đưa ra cách giải quyết cụ thể (liên hệ phòng ban phù hợp hoặc kênh hỗ trợ chính thức)
                - Nếu có thể, gợi ý loại đơn vị cần liên hệ dựa theo trường đại học bạn đang tư vấn (ví dụ: Phòng Tổ chức Hành chính, Phòng Đào tạo...)
                === NGUYÊN TẮC BẮT BUỘC ===
                - TUYỆT ĐỐI không suy diễn thông tin từ ngữ cảnh.
                - TUYỆT ĐỐI không trả lời theo nội dung ngữ cảnh nếu không khớp rõ ràng.
                - Không bịa thông tin.
                - Không cố gắng “trả lời cho có”.
                - Nếu câu hỏi vẫn thuộc phạm vi tư vấn tuyển sinh nhưng thiếu thông tin, hãy lịch sự yêu cầu người dùng cung cấp thêm dữ liệu cần thiết(thay vì từ chối trả lời).

                === VIỆC BẠN PHẢI LÀM ===
                1. Nhận diện rằng nội dung hiện có KHÔNG trả lời đúng câu hỏi.
                2. Phản hồi một cách lịch sự, rõ ràng, không máy móc, tự nhiên như 1 tư vấn tuyển sinh
                3. Hướng người dùng đi đúng hướng tiếp theo.
                4. Có thể chào hỏi nếu người dùng gửi lời chào
                5. Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
                6. Giải thích rằng hệ thống hiện chưa có dữ liệu phù hợp 
                === PHONG CÁCH TRẢ LỜI ===
                - Thân thiện, tự nhiên, không máy móc
                - Không chào hỏi dài dòng
                - Trả lời theo định dạng Markdown: dùng tiêu đề ##, gạch đầu dòng -, xuống dòng rõ ràng.
                """
                full_response = ""
                async for chunk in self.llm.astream(prompt):
                    text = chunk.content or ""
                    full_response += text
                    yield text
                    await asyncio.sleep(0)  # Nhường event loop

                memory.save_context({"input": query}, {"output": full_response})  
                print("Saved to memory. Current messages:", len(memory.chat_memory.messages))

                # === Lưu bot response vào DB ===
                bot_msg = ChatInteraction(
                    message_text=full_response,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=True,
                    sender_id=None,
                    session_id=session_id
                )
                db.add(bot_msg)
                db.flush()
                # 🧩 5. Commit 1 lần duy nhất
                db.commit()
                self.update_faq_statistics(db, bot_msg.interaction_id, intent_id=intent_id)
                self.update_faq_statistics_for_query(db, user_msg.interaction_id, intent_id = intent_id)
                print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        except Exception as e:
                print(f"response NA error: {e}")
        finally:
            db.close() 

    def add_interaction_and_faq_for_intent_0(self, full_response: str, session_id: int = 1, user_id: int = 1, intent_id: int = 1, message: str = ""):
            db = SessionLocal()
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            else:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=user_id,
                    session_id=session_id
                )
                db.add(user_msg)
                db.flush()
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id
            )
            db.add(bot_msg)
            db.flush()
            db.commit()
            self.update_faq_statistics_for_query(db, user_msg.interaction_id, intent_id = intent_id)



    def update_faq_statistics_for_query(self, db: Session, query_id: int, intent_id: int = 1):
        
        try:
            response = db.query(ChatInteraction).filter(
            ChatInteraction.interaction_id == query_id,
            ChatInteraction.is_from_bot == False
        ).first()

            if not response:
                raise ValueError("Chatbot response not found")

            faq = FaqStatistics(
                query_from_user_id = query_id,
                intent_id = intent_id
            )
            db.add(faq)
            db.commit()

        except Exception as e:
            db.rollback()
            print(f"Error updating FaqStatistics: {e}")

    def create_training_qa(self, db: Session, intent_id: int, question: str, answer: str, target_audiences: List[str], created_by: int):
        qa = TrainingQuestionAnswer(
            question=question,
            answer=answer,
            intent_id=intent_id,
            target_audiences=target_audiences,
            created_by=created_by,
            status="draft"
        )
        db.add(qa)
        db.commit()
        db.refresh(qa)

        return qa

    def approve_training_qa(self, db: Session, qa_id: int, reviewer_id: int):
        qa = db.query(TrainingQuestionAnswer).filter_by(question_id=qa_id).first()
        if not qa:
            raise Exception("QA not found")

        if qa.status != "draft":
            raise Exception("Only draft QA can be approved")
        
        intent = db.query(Intent).filter_by(intent_id=qa.intent_id).first()
        audience_names = qa.target_audiences or []
        audiences = db.query(TargetAudience).filter(
        TargetAudience.name.in_(audience_names)
        ).all()
        if not audiences:
            raise Exception("No valid audiences found")
        audience_ids = [a.id for a in audiences]
        audience_names = [a.name for a in audiences]
        filtered_audience_names = [a.present_name for a in audiences]
        
        # embed question (answer không embed)
        embedding = self.embeddings.embed_query(qa.question)
        point_id = str(uuid.uuid4())

        # push to Qdrant
        self.qdrant_client.upsert(
            collection_name="training_qa",
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question_id": qa.question_id,
                        "intent_id": qa.intent_id,
                        "intent_name": intent.intent_name if intent else None,
                        #MULTI AUDIENCE
                        "audience_ids": audience_ids,
                        "audience_names": filtered_audience_names,
                        "question_text": qa.question,
                        "answer_text": qa.answer,
                        "type": "training_qa"
                    }
                )
            ]
        )

        # update DB
        qa.status = "approved"
        qa.approved_by = reviewer_id
        qa.approved_at = datetime.now().date()  # Convert datetime to date
        db.commit()

        return {
            "postgre_question_id": qa.question_id,
            "qdrant_question_id": point_id
        }

    def delete_training_qa(self, db: Session, qa_id: int):
        
        qa = db.query(TrainingQuestionAnswer).filter_by(question_id=qa_id).first()
        if not qa:
            raise Exception("Training QA not found")

        # Xóa vector trong Qdrant
        self.qdrant_client.delete(
            collection_name="training_qa",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="question_id",
                            match=models.MatchValue(value = qa_id)
                        )
                    ]
                )
            )
        )

        # Xóa trong DB
        db.delete(qa)
        db.commit()

        return {"deleted_question_id": qa_id}

    def create_document(self, db: Session, title: str, file_path: str, intend_id: int, target_audiences: List[str], created_by: int):
        new_doc = KnowledgeBaseDocument(
            title=title,
            file_path=file_path,
            intend_id=intend_id,
            target_audiences=target_audiences,
            status="draft",
            created_by=created_by,
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        return new_doc

    def approve_document(self, db: Session, document_id: int, reviewer_id: int, intent_id: int, metadata: dict = None):

        doc = db.query(KnowledgeBaseDocument).filter_by(document_id=document_id).first()
        if not doc:
            raise Exception("Document not found")

        if doc.status != "draft":
            raise Exception("Only draft documents can be approved")
        audience_names_input = doc.target_audiences or []

        audiences = db.query(TargetAudience).filter(
            TargetAudience.name.in_(audience_names_input)
        ).all()

        if not audiences:
            raise Exception("No valid audiences found")

        audience_ids = [a.id for a in audiences]
        audience_names = [a.name for a in audiences]
        filtered_audience_names = [a.present_name for a in audiences]
        missing = set(audience_names_input) - set(audience_names)
        if missing:
            raise Exception(f"Audience not found: {missing}")
        
        abs_path = os.path.abspath(doc.file_path)
        print("OPEN FILE:", abs_path)
        intent = db.query(Intent).filter_by(intent_id=intent_id).first()
        with open(abs_path, "rb") as f:
            file_bytes = f.read()

        # 3. Detect MIME type từ extension (DocumentProcessor cần)
        mime_map = {
            ".pdf":  "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc":  "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls":  "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".txt":  "text/plain",
        }
        ext = os.path.splitext(doc.file_path)[1].lower()
        mime_type = mime_map.get(ext, "text/plain")
        content = DocumentProcessor.extract_text(
        file_content=file_bytes,
        filename=os.path.basename(doc.file_path),
        mime_type=mime_type
        )
        # --- Split text ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(content)

        qdrant_ids = []

        # --- Save chunks to DB & Qdrant ---
        for i, chunk in enumerate(chunks):

            # # Save DocumentChunk in DB
            # db_chunk = DocumentChunk(
            #     chunk_text=chunk,
            #     document_id=document_id,
            #     created_by=reviewer_id
            # )
            # db.add(db_chunk)
            # db.flush()   # get chunk_id

            # Embed
            embedding = self.embeddings.embed_query(chunk)
            point_id = str(uuid.uuid4())

            # Push to Qdrant
            self.qdrant_client.upsert(
                collection_name="knowledge_base_documents",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_id": document_id,
                            "chunk_index": i,
                            "chunk_text": chunk,
                            # multi audience
                            "audience_ids": audience_ids,
                            "audience_names": filtered_audience_names,
                            "intent_id": intent_id,
                            "intent_name": intent.intent_name if intent else None,
                            "metadata": metadata or {},
                            "type": "document"
                        }
                    )
                ]
            )

            qdrant_ids.append(point_id)

        # update document status
        doc.status = "approved"
        doc.reviewed_by = reviewer_id
        doc.reviewed_at = datetime.now().date()  # Convert datetime to date
        db.commit()

        return {
            "document_id": document_id,
            "status": doc.status
        }

    def delete_document(self, db: Session, document_id: int):
        doc = db.query(KnowledgeBaseDocument).filter_by(document_id=document_id).first()
        if not doc:
            raise Exception("Document not found")

        # Xóa sạch vector trong Qdrant
        self.qdrant_client.delete(
            collection_name="knowledge_base_documents",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value = document_id)
                        )
                    ]
                )
            )
        )

        # Xóa chunks trong DB
        dl = db.query(DocumentChunk).filter_by(document_id=document_id)
        if dl:
            dl.delete()
        # Xóa document trong DB
        db.delete(doc)
        db.commit()

        return {"deleted_document_id": document_id}
    



    # def add_document(self, document_id: int, content: str, intend_id: int, metadata: dict = None):
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,      # Size optimal cho Vietnamese
    #         chunk_overlap=200     # Overlap to preserve context
    #     )
    #     chunks = text_splitter.split_text(content)
        
    #     chunk_ids = []
    #     for i, chunk in enumerate(chunks):
    #         # Embed chunk
    #         embedding = self.embeddings.embed_query(chunk)
    #         point_id = str(uuid.uuid4())
            
    #         # Upsert to Qdrant
    #         self.qdrant_client.upsert(
    #             collection_name="knowledge_base_documents",
    #             points=[
    #                 PointStruct(
    #                     id=point_id,
    #                     vector=embedding,
    #                     payload={
    #                         "document_id": document_id,
    #                         "chunk_index": i,
    #                         "chunk_text": chunk,
    #                         "intend_id": intend_id,
    #                         "metadata": metadata or {},
    #                         "type": "document"
    #                     }
    #                 )
    #             ]
    #         )
    #         chunk_ids.append(point_id)
        
    #     return chunk_ids
    
    def cross_scope_search(self, query: str, top_k: int = 3):
        query_embedding = self.embeddings.embed_query(query)

        # ===== 1. SEARCH TRAINING QA =====
        qa_results = self.qdrant_client.search(
            collection_name=self.training_qa_collection,
            query_vector=query_embedding,
            limit=top_k
        )
        
        if qa_results and qa_results[0].score >= 0.5:
            top = qa_results[0]
           
            return {
                "source": "training_qa",
                "audience_ids": top.payload.get("audience_ids"),
                "audience_names": top.payload.get("audience_names"),
                "intent_id": top.payload.get("intent_id"),
                "intent_name": top.payload.get("intent_name"),
                "question": top.payload.get("question_text"),
                "score": top.score
            }

        # ===== 2. SEARCH DOCUMENT =====
        doc_results = self.qdrant_client.search(
            collection_name=self.documents_collection,
            query_vector=query_embedding,
            limit=top_k
        )

        if doc_results and doc_results[0].score >= 0.5:
            top = doc_results[0]

            return {
                "source": "document",
                "audience_ids": top.payload.get("audience_ids"),
                "audience_names": top.payload.get("audience_names"),
                "intent_id": top.payload.get("intent_id"),
                "intent_name": top.payload.get("intent_name"),
                "chunk_preview": top.payload.get("chunk_text", "")[:200],
                "score": top.score
            }

        return None

    def add_training_qa(self, db: Session, intent_id: int, question_text: str, answer_text: str):
        """
        Add training Q&A pair vào Qdrant
        
        Chỉ embed question, không embed answer:
        - Answer stored ở DB, retrieve khi match found
        - Question dùng để search/match
        - Tiết kiệm storage, tăng search speed
        
        Args:
            question_id: Primary key của training Q&A
            intent_id: Intent này thuộc intent nào
            question_text: Question để embed
            answer_text: Answer (lưu ở DB, không embed)
        
        Returns:
            embedding_id: Qdrant point ID
        """
        new_qa = TrainingQuestionAnswer(
            question=question_text,
            answer=answer_text,
            intent_id=1,
            created_by=1,
            status='draft'  # New Q&A starts as draft, needs review before training
        )
        db.add(new_qa)
        db.commit()
        db.refresh(new_qa)
        # Embed question text
        embedding = self.embeddings.embed_query(question_text)
        point_id = str(uuid.uuid4())
        
        # Upsert vào training_qa collection
        # Metadata:
        # - question_id: Link về DB
        # - intent_id: Để track intent stats
        # - question_text: Lưu original text (optional, space saving)
        # - answer_text: Lưu answer (retrieve khi match)
        self.qdrant_client.upsert(
            collection_name=self.training_qa_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question_id": new_qa.question_id,
                        "intent_id": intent_id,
                        "question_text": question_text,
                        "answer_text": answer_text,
                        "type": "training_qa"
                    }
                )
            ]
        )
        
        return {
            "postgre_question_id": new_qa.question_id,
            "qdrant_question_id": point_id
        }
    
    


    def search_documents(
        self,
        query: str,
        audience_ids: int,
        intent_id: int = None,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        stage: str = "document_search",
    ):

        """
        Search documents (Fallback)
        
        Fallback path: Tìm document chunks khi training Q&A không match
        - Query → Embed → Search documents collection
        - Return top_k chunks
        - LLM sẽ synthesize answer từ chunks
        
        Args:
            query: User question
            top_k: Số chunks (lower score → fallback)
        
        Returns:
            List of document chunks
        """

        must_conditions = [
            {
                "key": "audience_ids",
                "match": {"value": audience_ids}
            }
        ]
        if intent_id:
            must_conditions.append({
                "key": "intent_id",
                "match": {"value": intent_id}
            })

        
        start = time.perf_counter()
        self._debug_log(
            f"{stage}: start search_documents top_k={top_k} query_len={len(query or '')}",
            trace_id
        )

        try:
            query_embedding = self.embeddings.embed_query(query)

            results = self.qdrant_client.search(
                collection_name=self.documents_collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter={
                    "must": must_conditions
                }
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            top_score = float(getattr(results[0], "score", 0.0)) if results else 0.0
            top_payload = getattr(results[0], "payload", {}) if results else {}
            top_document_id = (top_payload or {}).get("document_id")
            self._debug_log(
                f"{stage}: success results={len(results)} top_score={top_score:.6f} "
                f"top_document_id={top_document_id} elapsed_ms={elapsed_ms}",
                trace_id
            )
            return results
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._debug_log(
                f"{stage}: search_documents error type={type(e).__name__} "
                f"message={e} elapsed_ms={elapsed_ms}",
                trace_id
            )
            print(f"Qdrant search_documents timeout/error: {e}")
            return []

    def _fetch_document_names_by_id(self, document_ids: List[int]) -> Dict[int, str]:
        """
        Fetch document display names from DB by document_id.
        """
        if not document_ids:
            return {}

        db = SessionLocal()
        try:
            rows = (
                db.query(KnowledgeBaseDocument.document_id, KnowledgeBaseDocument.title)
                .filter(KnowledgeBaseDocument.document_id.in_(document_ids))
                .all()
            )
            return {int(row.document_id): row.title for row in rows}
        except Exception:
            return {}
        finally:
            db.close()

    def extract_document_sources(self, results: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract, dedupe, and enrich citation sources with document names.
        """
        document_ids: List[int] = []
        seen = set()

        for result in results or []:
            payload = getattr(result, "payload", {}) or {}
            document_id = payload.get("document_id")
            if document_id is None:
                continue
            try:
                normalized_id = int(document_id)
            except (TypeError, ValueError):
                continue

            if normalized_id in seen:
                continue
            seen.add(normalized_id)
            document_ids.append(normalized_id)

        name_map = self._fetch_document_names_by_id(document_ids)
        return [
            {
                "document_id": document_id,
                "file_name": name_map.get(document_id),
            }
            for document_id in document_ids
        ]

    def build_document_search_result(self, doc_results: List[Any]) -> Dict[str, Any]:
        """
        Build a stable document-search response object used by chat pipelines.
        """
        top_match = doc_results[0] if doc_results else None
        intent_id = 0
        confidence = 0.0
        audience_ids = 0
        audience_names = ""
        try:
            if top_match is not None:
                payload = getattr(top_match, "payload", {}) or {}
                intent_id = payload.get("intent_id") or 0
                confidence = float(getattr(top_match, "score", 0.0) or 0.0)
                audience_ids = top_match.payload.get("audience_ids"),
                audience_names = top_match.payload.get("audience_names"),
        except Exception as e:
                print(f"build_document_search_result error: {e}")
        return {
            "response": doc_results,
            "response_source": "document",
            "confidence": confidence,
            "top_match": top_match,
            "audience_ids": audience_ids,
            "audience_names": audience_names,
            "intent_id": intent_id,
            "sources": self.extract_document_sources(doc_results),
        }

    async def infer_used_document_ids(
        self,
        query: str,
        answer_text: str,
        context_chunks: List[Any],
        allowed_sources: List[Dict[str, Any]],
        trace_id: Optional[str] = None,
    ) -> List[int]:
        """
        Ask LLM to identify which retrieved document IDs were actually used in the answer.
        Final output is always constrained to allowed source IDs (anti-hallucination guard).
        """
        allowed_ids = []
        for src in allowed_sources or []:
            try:
                doc_id = int(src.get("document_id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if doc_id not in allowed_ids:
                allowed_ids.append(doc_id)

        if not allowed_ids:
            return []

        snippet_lines = []
        for idx, chunk in enumerate(context_chunks[:12], start=1):
            payload = getattr(chunk, "payload", {}) or {}
            raw_doc_id = payload.get("document_id")
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError):
                continue
            if doc_id not in allowed_ids:
                continue
            chunk_text = (payload.get("chunk_text") or "").strip().replace("\n", " ")
            snippet_lines.append(f"{idx}. [DOC_ID={doc_id}] {chunk_text[:500]}")

        context_for_citation = "\n".join(snippet_lines)
        prompt = f"""
Bạn là bộ máy trích dẫn nguồn.
Nhiệm vụ: xác định tài liệu nào THỰC SỰ được dùng để tạo câu trả lời.

Allowed document IDs: {allowed_ids}
Question: {query}
Answer: {answer_text}

Retrieved context snippets:
{context_for_citation}

Yêu cầu:
- Chỉ chọn ID nằm trong allowed IDs.
- Nếu không đủ bằng chứng thì trả [].
- Trả về DUY NHẤT JSON array số nguyên, ví dụ: [58, 60] hoặc [].
"""

        try:
            res = await self.llm.ainvoke(prompt)
            raw = (res.content or "").strip()
            parsed_ids: List[int] = []
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    for x in obj:
                        try:
                            parsed_ids.append(int(x))
                        except (TypeError, ValueError):
                            continue
            except Exception:
                parsed_ids = [int(x) for x in re.findall(r"\d+", raw)]

            allowed_set = set(allowed_ids)
            final_ids = []
            for doc_id in parsed_ids:
                if doc_id in allowed_set and doc_id not in final_ids:
                    final_ids.append(doc_id)

            self._debug_log(
                f"citation_guard: allowed={allowed_ids} raw='{raw}' final={final_ids}",
                trace_id
            )
            return final_ids
        except Exception as e:
            self._debug_log(
                f"citation_guard: error type={type(e).__name__} message={e}",
                trace_id
            )
            return []

    def is_insufficient_answer(self, answer_text: str) -> bool:
        """
        Detect generic "insufficient information" answers.
        If true, citations should be hidden to avoid misleading evidence.
        """
        text = (answer_text or "").strip().lower()
        if not text:
            return True

        markers = [
            "không đủ thông tin",
            "không có đủ thông tin",
            "chưa đủ thông tin",
            "không có thông tin",
            "chưa có thông tin",
            "không tìm thấy thông tin",
            "không có dữ liệu",
            "chưa có dữ liệu",
            "không thể xác định",
            "không thể trả lời",
            "không thể cung cấp",
            "không rõ",
        ]
        return any(marker in text for marker in markers)
    

    def search_training_qa(
        self,
        query: str,
        audience_ids: int,
        intent_id: int = None,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        stage: str = "training_qa_search",
    ):

        """
        Search training Q&A (Priority 1)
        
        Fast path: Tìm pre-approved answers
        - Query → Embed → Search training_qa collection
        - Return top_k matches
        - filter score > 0.8
        
        Args:
            query: User question
            top_k: Số results (default 5)
        
        Returns:
            List of search results with scores
        """

        must_conditions = [
            {
                "key": "audience_ids",
                "match": {"value": audience_ids}
            }
        ]
        if intent_id:
            must_conditions.append({
                "key": "intent_id",
                "match": {"value": intent_id}
            })

        
        start = time.perf_counter()
        self._debug_log(
            f"{stage}: start search_training_qa top_k={top_k} query_len={len(query or '')}",
            trace_id
        )

        try:
            query_embedding = self.embeddings.embed_query(query)

            results = self.qdrant_client.search(
                collection_name=self.training_qa_collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter={
                    "must": must_conditions
                }
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            top_score = float(getattr(results[0], "score", 0.0)) if results else 0.0
            top_payload = getattr(results[0], "payload", {}) if results else {}
            top_question_id = (top_payload or {}).get("question_id")
            self._debug_log(
                f"{stage}: success results={len(results)} top_score={top_score:.6f} "
                f"top_question_id={top_question_id} elapsed_ms={elapsed_ms}",
                trace_id
            )
            return results
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._debug_log(
                f"{stage}: search_training_qa error type={type(e).__name__} "
                f"message={e} elapsed_ms={elapsed_ms}",
                trace_id
            )
            print(f"Qdrant search_training_qa timeout/error: {e}")
            return []


    def hybrid_search(self, audience_ids: int, query: str, intent_id: int = None, trace_id: Optional[str] = None):

        """
        Hybrid RAG Search Strategy
        
        PRIORITY SYSTEM (Cascade):
        1. TIER 1 - Training Q&A (score > 0.8)
           - Highest confidence, direct answer
           - No LLM needed, fast response
           
        2. TIER 2 - Training Q&A (0.7 < score <= 0.8)
           - Good match but not perfect
           - Use as primary answer + add document context
           
        3. TIER 3 - Document Search + LLM Generation
           - No training Q&A match
           - Search documents, LLM synthesize
           - Lower confidence, show sources
           
        4. TIER 4 - Fallback
           - Nothing found
           - Suggest live chat with officer
        
        Returns:
            {
                "response": str,
                "response_source": "training_qa" | "document" | "fallback",
                "confidence": float,
                "top_match": obj,
                "intent_id": int,
                "sources": list
            }
        """
        
        # STEP 1: Search training Q&A

        self._debug_log(
            f"hybrid_search: start query_len={len(query or '')}",
            trace_id
        )
        qa_results = self.search_training_qa(
            query,
            audience_ids, 
            intent_id,
            top_k=3,
            trace_id=trace_id,
            stage="hybrid_training_qa_search",
        )
        if qa_results:
            print("qa result " + qa_results[0].payload.get("answer_text"))
            print(f"score: + {qa_results[0].score}")
            self._debug_log(
                f"hybrid_search: qa_results={len(qa_results)} top_score={qa_results[0].score:.6f}",
                trace_id
            )
        else:
            self._debug_log("hybrid_search: qa_results=0", trace_id)

        # TIER 1: Perfect match (score > 0.7)
        if qa_results and qa_results[0].score >= 0.5:
            top_match = qa_results[0]
            self._debug_log("hybrid_search: tier=training_qa", trace_id)
            return {
                "response_official_answer": top_match.payload.get("answer_text"),
                "response_source": "training_qa",
                "confidence": top_match.score,
                "top_match": top_match,
                "intent_id": top_match.payload.get("intent_id"),
                "audience_ids": top_match.payload.get("audience_ids"),
                "audience_names": top_match.payload.get("audience_names"),
                "question_id": top_match.payload.get("question_id"),
                "sources": []
            }
        
        
        # TIER 2: No training Q&A match, try documents

        doc_results = self.search_documents(
            query,
            audience_ids, 
            intent_id,
            top_k=5,
            trace_id=trace_id,
            stage="hybrid_tier2_document_search",
        )
        result = self.build_document_search_result(doc_results)
        self._debug_log(
            f"hybrid_search: tier=document confidence={result.get('confidence', 0.0):.6f} "
            f"sources={len(result.get('sources', []))}",
            trace_id
        )
        return result

        
    def _get_user_personality_and_academics(self, user_id: int, db: Session) -> Dict[str, Any]:
        out = {
            "personality_summary": None,
            "riasec": None,
            "academic_summary": None,
            "gpa": None,
            "subjects": {}
        }

        # --- RIASEC result ---
        ri = (
            db.query(RiasecResult)
            .filter(RiasecResult.customer_id == user_id)
            .order_by(RiasecResult.result_id.desc())
            .first()
        )

        if ri:
            out["riasec"] = {
                "R": ri.score_realistic,
                "I": ri.score_investigative,
                "A": ri.score_artistic,
                "S": ri.score_social,
                "E": ri.score_enterprising,
                "C": ri.score_conventional,
            }
            # `result` field = summary của bạn
            out["personality_summary"] = ri.result or self._riasec_to_summary(out["riasec"])

        # --- Academic scores ---
        score = (
            db.query(AcademicScore)
            .filter(AcademicScore.customer_id == user_id)
            .first()
        )

        if score:
            subj_map = {
            "math": score.math,
            "literature": score.literature,
            "english": score.english,
            "physics": score.physics,
            "chemistry": score.chemistry,
            "biology": score.biology,
            "history": score.history,
            "geography": score.geography,
        }

            # simple GPA = average score
            valid_scores = [v for v in subj_map.values() if v is not None]
            gpa = round(sum(valid_scores) / len(valid_scores), 2)

            out["subjects"] = subj_map
            out["gpa"] = gpa
            out["academic_summary"] = (
                f"GPA xấp xỉ {gpa}. Các môn: " +
                ", ".join([f"{k}: {v}" for k, v in subj_map.items()])
            )
            print(out["academic_summary"])
        return out

    def _riasec_to_summary(self, ri_map: Dict[str,int]) -> str:
        # very small helper - bạn có thể mở rộng
        order = sorted(ri_map.items(), key=lambda x: -x[1])
        top = order[0][0] if order else None
        return f"Ưu thế RIASEC: {', '.join([f'{k}={v}' for k,v in ri_map.items()])}. Chính: {top}."

    def _get_all_majors_from_db(self, db: Session, limit: int = 200) -> List[Dict[str,Any]]:
        """
        Lấy danh sách majors
        """
        rows = db.query(Major).order_by(Major.major_name).limit(limit).all()
        majors = []
        for r in rows:
            majors.append({
                "major_id": r.major_id,
                "major_name": r.major_name,
            })
        return majors

    def _get_all_majors_and_specialization_from_db(self, db: Session, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Lấy danh sách majors kèm theo danh sách specializations
        """
        rows = (
            db.query(Major)
            .order_by(Major.major_name)
            .limit(limit)
            .all()
        )

        majors = []
        for r in rows:
            majors.append({
                "major_id": r.major_id,
                "major_name": r.major_name,
                "specializations": [
                    {
                        "specialization_id": s.specialization_id,
                        "specialization_name": s.specialization_name
                    }
                    for s in r.specializations
                ]
            })

        return majors
    
    

    

langchain_service = TrainingService()
