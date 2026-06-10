from datetime import datetime
from typing import Any, Dict, List, Optional
import time
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.chat_models import init_chat_model
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from sentence_transformers import CrossEncoder
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import asyncio
import httpx
from sqlalchemy.orm import Session
from app.models import schemas
from app.models.entities import (
    AcademicScore,
    ChatInteraction,
    ChatSession,
    DocumentChunk,
    FaqStatistics,
    Intent,
    KnowledgeBaseDocument,
    Major,
    ParticipateChatSession,
    RiasecResult,
    TargetAudience,
    TrainingQuestionAnswer,
    Users,
)
from app.models.database import SessionLocal
from sqlalchemy.exc import SQLAlchemyError
from app.services.memory_service import MemoryManager
from app.utils.document_processor import DocumentProcessor

memory_service = MemoryManager()
load_dotenv()
# print("Đang nạp mô hình Reranker lên RAM, vui lòng đợi vài giây...")
# RERANKER_MODEL = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
# print("Nạp mô hình thành công! Server sẵn sàng.")


class TrainingService:
    def __init__(self):
        self.top_k = os.getenv("TOP_K", 5)
        self.ai_api_key = os.getenv("AI_API_KEY")
        self.openai_llm = init_chat_model(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            api_key=self.ai_api_key,  # Truyền rõ ràng api key ở đây
            temperature=float(os.getenv("OPENAI_LLM_TEMPERATURE", 0.2)),
        )
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            api_key=self.ai_api_key,
        )
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if llm_provider == "gemini" and gemini_api_key:
            self.control_llm = ChatGoogleGenerativeAI(
                model=os.getenv(
                    "GEMINI_CONTROL_MODEL",
                    os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                ),
                google_api_key=gemini_api_key,
                temperature=float(os.getenv("GEMINI_CONTROL_TEMPERATURE", 0)),
            )
            self.answer_llm = ChatGoogleGenerativeAI(
                model=os.getenv(
                    "GEMINI_ANSWER_MODEL",
                    os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                ),
                google_api_key=gemini_api_key,
                temperature=float(os.getenv("GEMINI_ANSWER_TEMPERATURE", 0.3)),
            )
            print(
                "Gemini LLM enabled: control="
                f"{os.getenv('GEMINI_CONTROL_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))}, "
                "answer="
                f"{os.getenv('GEMINI_ANSWER_MODEL', os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'))}"
            )
        elif llm_provider == "openai":
            self.control_llm = init_chat_model(
                model=os.getenv(
                    "OPENAI_CONTROL_MODEL",
                    os.getenv("LLM_MODEL", "gpt-4.1-mini"),
                ),
                api_key=self.ai_api_key,
                temperature=float(os.getenv("OPENAI_CONTROL_TEMPERATURE", 0)),
            )
            self.answer_llm = init_chat_model(
                model=os.getenv(
                    "OPENAI_ANSWER_MODEL",
                    os.getenv("LLM_MODEL", "gpt-4.1-mini"),
                ),
                api_key=self.ai_api_key,
                temperature=float(os.getenv("OPENAI_ANSWER_TEMPERATURE", 0.3)),
            )
            print(
                "OpenAI LLM enabled: control="
                f"{os.getenv('OPENAI_CONTROL_MODEL', os.getenv('LLM_MODEL', 'gpt-4.1-mini'))}, "
                "answer="
                f"{os.getenv('OPENAI_ANSWER_MODEL', os.getenv('LLM_MODEL', 'gpt-4.1-mini'))}"
            )
        else:
            self.control_llm = self.openai_llm
            self.answer_llm = self.openai_llm
            print(
                f"WARNING: unsupported LLM_PROVIDER='{llm_provider}' or GEMINI_API_KEY is not set. "
                "Falling back to OpenAI LLM; "
                "OpenAI embeddings are still active."
            )

        # Backward-compatible alias for any older call sites.
        self.llm = self.control_llm

        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            timeout=30,
            check_compatibility=False,
        )
        self.async_qdrant_client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            timeout=40,
            check_compatibility=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self.training_qa_collection = "training_qa"
        self.documents_collection = "knowledge_base_documents"
        self.university_name = os.getenv(
            "CHAT_UNIVERSITY_NAME", "Đại học Giao thông Vận tải"
        )

        self._init_collections()

    def _init_collections(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.training_qa_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
        except:
            pass

        try:
            self.qdrant_client.create_collection(
                collection_name=self.documents_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
        except:
            pass

    def _debug_log(self, message: str, trace_id: Optional[str] = None):
        if trace_id:
            print(f"[RAG][{trace_id}] {message}")
            return
        print(f"[RAG] {message}")

    @staticmethod
    def _message_text(message_or_content: Any) -> str:
        """
        Normalize LangChain message content across OpenAI/Gemini providers.
        Gemini may return content as a list of text parts instead of a string.
        """
        content = getattr(message_or_content, "content", message_or_content)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None)
                    if text is not None:
                        parts.append(str(text))
            return "".join(parts)
        return str(content)

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
            session = ChatSession(session_type=session_type, start_time=datetime.now())
            db.add(session)
            db.flush()
            db.commit()
            return session.chat_session_id
        try:
            session = ChatSession(session_type=session_type, start_time=datetime.now())
            db.add(session)
            db.flush()

            # Add user vào participate table
            participate = ParticipateChatSession(
                user_id=user_id, session_id=session.chat_session_id
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
            interactions = (
                db.query(ChatInteraction)
                .filter(ChatInteraction.session_id == session_id)
                .order_by(ChatInteraction.timestamp.asc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "message_text": i.message_text,
                    "timestamp": i.timestamp.isoformat() if i.timestamp else None,
                    "is_from_bot": i.is_from_bot,
                    "rating": i.rating,
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
            sessions = (
                db.query(ChatSession)
                .filter(ChatSession.session_type == "chatbot")
                .join(ParticipateChatSession)
                .filter(ParticipateChatSession.user_id == user_id)
                .order_by(ChatSession.start_time.desc())
                .all()
            )

            result = []
            for session in sessions:
                # Lấy message cuối cùng làm preview
                last_msg = (
                    db.query(ChatInteraction)
                    .filter(ChatInteraction.session_id == session.chat_session_id)
                    .order_by(ChatInteraction.timestamp.desc())
                    .first()
                )

                result.append(
                    {
                        "session_id": session.chat_session_id,
                        "session_type": session.session_type,
                        "start_time": (
                            session.start_time.isoformat()
                            if session.start_time
                            else None
                        ),
                        "last_message_preview": (
                            last_msg.message_text[:50] + "..." if last_msg else ""
                        ),
                        "last_message_time": (
                            last_msg.timestamp.isoformat()
                            if last_msg and last_msg.timestamp
                            else None
                        ),
                    }
                )

            return result
        finally:
            db.close()

    def delete_chat_session(
        self, session_id: int, user_id: Optional[int] = None
    ) -> bool:
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

            session = query.filter(ChatSession.chat_session_id == session_id).first()

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
        print(f"lich su chat: {chat_history}")
        prompt = f"""
        Nhiệm vụ của bạn là VIẾT LẠI câu hỏi mới nhất của người dùng thành một truy vấn ĐỘC LẬP, NGẮN GỌN và ĐẦY ĐỦ NGỮ NGHĨA để tra cứu trong Vector Database.

        Cuộc hội thoại gần đây:
        {chat_history}

        Câu hỏi phản hồi mới nhất của người dùng:
        "{user_message}"

        NHIỆM VỤ:
        Viết lại câu hỏi mới nhất thành một câu truy vấn ĐỘC LẬP, ĐẦY ĐỦ NGỮ NGHĨA để máy tìm kiếm (Vector Search) có thể hiểu được chính xác mà không cần đọc lại lịch sử.
        === TỪ ĐIỂN ĐỒNG NGHĨA BẮT BUỘC ===
        Khi người dùng dùng các cụm từ sau, BẮT BUỘC mở rộng query để bao gồm các từ tương đương:
        1. Nhân sự / Con người
        - "cán bộ" / "cán bộ nhà trường" 
        → bao gồm: viên chức, giảng viên, nhân viên, người lao động

        - "giảng viên" 
        → bao gồm: nhà giáo, giáo viên, giảng viên đại học, viên chức giảng dạy

        - "nhân viên" 
        → bao gồm: viên chức, người lao động, cán bộ hành chính

        - "sinh viên" 
        → bao gồm: người học, học viên, nghiên cứu sinh

        - "hiệu trưởng"
        → bao gồm: giám đốc phân hiệu, ban giám hiệu, lãnh đạo trường

        - "trưởng khoa" / "phó trưởng khoa" / "lãnh đạo khoa"
        → bao gồm: ban lãnh đạo khoa, chủ nhiệm khoa, 
                người đứng đầu khoa
        → KHÔNG bao gồm: trưởng bộ môn, phó trưởng bộ môn
        (đây là cấp thấp hơn, chỉ tìm khi được hỏi trực tiếp)

        2. Hành chính / Chế độ
        - "nghỉ phép"
        → bao gồm: nghỉ hằng năm, nghỉ hè, nghỉ việc riêng, chế độ nghỉ, nghỉ không hưởng lương, phép năm

        - "lương"
        → bao gồm: tiền lương, thu nhập, thù lao, hệ số lương, mức lương, phụ cấp, lương 3P

        - "kỷ luật"
        → bao gồm: xử lý kỷ luật, hình thức kỷ luật, khiển trách, cảnh cáo, buộc thôi việc, vi phạm

        - "đánh giá viên chức"
        → bao gồm: xếp loại, phân loại viên chức, đánh giá cuối năm, hoàn thành nhiệm vụ

        3. Đào tạo / Học vụ
        - "học phí"
        → bao gồm: mức học phí, phí đào tạo, chi phí học tập, học phí tín chỉ, lệ 
        
        - "thời khóa biểu"
        → bao gồm: TKB, lịch học, kế hoạch giảng dạy, thời gian biểu, lịch dạy

        - "điểm thi"
        → bao gồm: kết quả học tập, điểm số, điểm học phần, bảng điểm, điểm tổng kết

        - "nghỉ học"
        → bao gồm: bảo lưu, nghỉ học tạm thời, tạm dừng học, hoãn học

        - "ra trường"
        → bao gồm: tốt nghiệp, xét tốt nghiệp, công nhận tốt nghiệp, nhận bằng, hoàn thành chương trình

        - "chương trình đào tạo"
        → bao gồm: CTĐT, giáo trình, khung chương trình, chuẩn đầu ra, học phần, tín chỉ

        - "tiếng anh" / "ngoại ngữ" / "anh văn"
        → bao gồm: chứng chỉ ngoại ngữ, chuẩn đầu ra ngoại ngữ, điều kiện ngoại ngữ, năng lực ngoại ngữ, TOEIC, IELTS, TOEFL, B1, B2, A2, khung năng lực ngoại ngữ 6 bậc, CEFR, chứng chỉ tiếng Anh, tiếng Anh đầu ra

        - "chuẩn đầu ra" / "điều kiện tốt nghiệp" / "điều kiện ra trường"
        → bao gồm: chuẩn đầu ra ngoại ngữ, chuẩn đầu ra tin học, điều kiện xét tốt nghiệp, yêu cầu tốt nghiệp, chứng chỉ bắt buộc, chứng chỉ đầu ra

        - "đồ án" / "luận văn" / "khóa luận"
        → bao gồm: đồ án tốt nghiệp, luận văn tốt nghiệp, đề tài tốt nghiệp, bảo vệ đồ án, hội đồng chấm đồ án
        4. Đơn vị / Tổ chức
        - "phòng ban"
        → bao gồm: đơn vị, bộ phận, khoa, bộ môn, trung tâm, ban

        - "ký túc xá"
        → bao gồm: KTX, nội trú, chỗ ở sinh viên, khu nội trú, nhà ở sinh viên, phòng kí túc xá

        5. Văn bản / Quy định
        - "quy định"
        → bao gồm: quy chế, nội quy, điều lệ, quy trình, hướng dẫn, thông tư, nghị định

        - "hồ sơ"
        → bao gồm: giấy tờ, tài liệu, đơn, văn bản, minh chứng, chứng từ

        - "xin / đăng ký"
        → bao gồm: nộp đơn, đề nghị, làm thủ tục, đăng ký, nộp hồ sơ

        Ví dụ:
        - "quy trình nghỉ phép đối với cán bộ" 
        → rewrite: "chế độ nghỉ phép của viên chức giảng viên nhân viên"
        - "lương của cán bộ" 
        → rewrite: "chế độ tiền lương viên chức giảng viên người lao động"
        HƯỚNG DẪN:
        1. KHÔI PHỤC NGỮ CẢNH: Thay thế các đại từ (nó, trường này, ngành đó), câu rút gọn, hoặc chủ ngữ bị khuyết bằng các DANH TỪ RIÊNG cụ thể (tên trường, tên cơ sở, tên ngành, phương thức) ĐÃ XUẤT HIỆN trong hội thoại trước đó.
        ĐẶC BIỆT: Các cụm "dẫn chứng này", "thông tin trên", "quy định đó", "điều vừa nói", 
        "câu trả lời trước" → BẮT BUỘC tra lại lịch sử để tìm CHỦ ĐỀ CỤ THỂ đang được nhắc đến 
        rồi thay thế vào.
        
        Ví dụ:
        - Lịch sử: Bot vừa trả lời "kế hoạch nhiệm vụ năm học lưu trữ 20 năm theo quy định Bộ GDĐT"
        - User hỏi: "dẫn chứng này nằm ở quy định nào, điều khoản nào"
        - Rewrite thành: "Quy định về thời hạn lưu trữ kế hoạch nhiệm vụ năm học 20 năm 
                            nằm ở văn bản pháp lý nào, điều khoản nào?"
        
        2. KHÔNG BỊA ĐẶT: TUYỆT ĐỐI KHÔNG thêm thông tin hoàn toàn mới chưa từng được nhắc đến. KHÔNG thay đổi mục tiêu câu hỏi.
        3. GIỮ NGUYÊN NẾU ĐÃ RÕ RÀNG: Nếu câu hỏi mới nhất đã tự mang đủ ngữ nghĩa độc lập, hãy trả về NGUYÊN VĂN.
        4. KẾT QUẢ ĐẦU RA: Chỉ in ra ĐÚNG 1 CÂU truy vấn tiếng Việt, TUYỆT ĐỐI KHÔNG giải thích, KHÔNG có dấu ngoặc kép bọc ngoài.
        5. QUY TẮC BẢO TOÀN TỪ KHÓA CƠ SỞ (CỰC KỲ QUAN TRỌNG):
            - NẾU Câu hỏi phản hồi mới nhất của người dùng CÓ chứa các từ khóa chỉ cơ sở (như "Phân hiệu", "UTC2", "TP.HCM", "Cơ sở chính", "Hà Nội"): BẮT BUỘC phải giữ lại y chang các từ khóa này cuối câu query.
            - NẾU câu hỏi gốc KHÔNG nhắc đến cơ sở nào: TUYỆT ĐỐI KHÔNG tự ý suy diễn hay thêm các từ "Phân hiệu", "UTC2", "mã GSA" vào câu query.
        6. QUY TẮC BẢO TOÀN TÊN VĂN BẢN:
            - NẾU câu hỏi gốc chứa mã/tên văn bản (như "QĐ1821", "Thông tư 08", "Quy chế đào tạo"):
            CHỈ giữ nguyên mã/tên đó. TUYỆT ĐỐI KHÔNG suy diễn nội dung, phạm vi áp dụng,
            tên đầy đủ, hay đơn vị ban hành của văn bản đó — dù bot đã đề cập đến văn bản
            này ở các lượt trả lời trước.
            - Ví dụ SAI: "QĐ1821" → "Quyết định 1821 về cơ sở vật chất của UTC2"
            - Ví dụ ĐÚNG: "QĐ1821" → giữ nguyên "QĐ1821"
        7. NẾU câu hỏi của người dùng là hỏi về thân phận, chức vụ của một người (Ví dụ: "Đặng Văn Ơn là ai?", "ThS Đặng Văn Ơn làm gì?"):
            BẮT BUỘC phải bọc tên người đó trong dấu ngoặc kép kép "".
            BẮT BUỘC bổ sung thêm các cụm từ: chức vụ, phòng ban
            Ví dụ: > - User hỏi: "ông đặng văn ơn có chức vụ gì"
            Bạn phải sinh ra Query: Tìm chức vụ, đơn vị công tác và phòng ban của "Đặng Văn Ơn"
        """
        # THÊM LOG NÀY
        print(f"[ENRICH] chat_history length={len(str(chat_history))}")
        print(f"[ENRICH] chat_history preview={str(chat_history)[:300]}")
        # assume async predict exists
        enriched = await self.control_llm.ainvoke(prompt)
        print("==== RAW RESPONSE ====")
        print(user_message)
        print("======================")
        # fallback: if empty use original
        enriched_lines = (
            self._message_text(enriched).strip().splitlines() if enriched else []
        )
        enriched_txt = enriched_lines[0] if enriched_lines else user_message
        print("==== ENRICH QUERY ====")
        print(enriched_txt)
        print("======================")
        return enriched_txt

    async def enrich_query_tuyensinh(self, session_id: str, user_message: str) -> str:
        memory = memory_service.get_memory(session_id)
        mem_vars = memory.load_memory_variables({})
        chat_history = mem_vars.get("chat_history", "")

        prompt = f"""
        Bạn là một trợ lý chuẩn hóa truy vấn cho chatbot RAG tư vấn tuyển sinh của {self.university_name}.

        Cuộc hội thoại gần đây:
        {chat_history}

        Phản hồi mới nhất của người dùng:
        "{user_message}"

        NHIỆM VỤ:
        === TỪ ĐIỂN ĐỒNG NGHĨA BẮT BUỘC ===
        Khi người dùng dùng các cụm từ sau, BẮT BUỘC mở rộng query để bao gồm các từ tương đương:
        1. Nhân sự / Con người
        - "cán bộ" / "cán bộ nhà trường" 
        → bao gồm: viên chức, giảng viên, nhân viên, người lao động

        - "giảng viên" 
        → bao gồm: nhà giáo, giáo viên, giảng viên đại học, viên chức giảng dạy

        - "nhân viên" 
        → bao gồm: viên chức, người lao động, cán bộ hành chính

        - "sinh viên" 
        → bao gồm: người học, học viên, nghiên cứu sinh

        - "hiệu trưởng"
        → bao gồm: giám đốc phân hiệu, ban giám hiệu, lãnh đạo trường

        - "trưởng khoa"
        → bao gồm: lãnh đạo khoa, chủ nhiệm khoa, trưởng bộ môn, người đứng đầu đơn vị

        2. Hành chính / Chế độ
        - "nghỉ phép"
        → bao gồm: nghỉ hằng năm, nghỉ hè, nghỉ việc riêng, chế độ nghỉ, nghỉ không hưởng lương, phép năm

        - "lương"
        → bao gồm: tiền lương, thu nhập, thù lao, hệ số lương, mức lương, phụ cấp, lương 3P

        - "kỷ luật"
        → bao gồm: xử lý kỷ luật, hình thức kỷ luật, khiển trách, cảnh cáo, buộc thôi việc, vi phạm

        - "đánh giá viên chức"
        → bao gồm: xếp loại, phân loại viên chức, đánh giá cuối năm, hoàn thành nhiệm vụ

        3. Đào tạo / Học vụ
        - "học phí"
        → bao gồm: mức học phí, phí đào tạo, chi phí học tập, học phí tín chỉ, lệ 
        
        - "thời khóa biểu"
        → bao gồm: TKB, lịch học, kế hoạch giảng dạy, thời gian biểu, lịch dạy

        - "điểm thi"
        → bao gồm: kết quả học tập, điểm số, điểm học phần, bảng điểm, điểm tổng kết

        - "nghỉ học"
        → bao gồm: bảo lưu, nghỉ học tạm thời, tạm dừng học, hoãn học

        - "ra trường"
        → bao gồm: tốt nghiệp, xét tốt nghiệp, công nhận tốt nghiệp, nhận bằng, hoàn thành chương trình

        - "chương trình đào tạo"
        → bao gồm: CTĐT, giáo trình, khung chương trình, chuẩn đầu ra, học phần, tín chỉ

        - "tiếng anh" / "ngoại ngữ" / "anh văn"
        → bao gồm: chứng chỉ ngoại ngữ, chuẩn đầu ra ngoại ngữ, điều kiện ngoại ngữ, năng lực ngoại ngữ, TOEIC, IELTS, TOEFL, B1, B2, A2, khung năng lực ngoại ngữ 6 bậc, CEFR, chứng chỉ tiếng Anh, tiếng Anh đầu ra

        - "chuẩn đầu ra" / "điều kiện tốt nghiệp" / "điều kiện ra trường"
        → bao gồm: chuẩn đầu ra ngoại ngữ, chuẩn đầu ra tin học, điều kiện xét tốt nghiệp, yêu cầu tốt nghiệp, chứng chỉ bắt buộc, chứng chỉ đầu ra

        - "đồ án" / "luận văn" / "khóa luận"
        → bao gồm: đồ án tốt nghiệp, luận văn tốt nghiệp, đề tài tốt nghiệp, bảo vệ đồ án, hội đồng chấm đồ án
        4. Đơn vị / Tổ chức
        - "phòng ban"
        → bao gồm: đơn vị, bộ phận, khoa, bộ môn, trung tâm, ban

        - "ký túc xá"
        → bao gồm: KTX, nội trú, chỗ ở sinh viên, khu nội trú, nhà ở sinh viên, phòng kí túc xá

        5. Văn bản / Quy định
        - "quy định"
        → bao gồm: quy chế, nội quy, điều lệ, quy trình, hướng dẫn, thông tư, nghị định

        - "hồ sơ"
        → bao gồm: giấy tờ, tài liệu, đơn, văn bản, minh chứng, chứng từ

        - "xin / đăng ký"
        → bao gồm: nộp đơn, đề nghị, làm thủ tục, đăng ký, nộp hồ sơ

        Ví dụ:
        - "quy trình nghỉ phép đối với cán bộ" 
        → rewrite: "chế độ nghỉ phép của viên chức giảng viên nhân viên"
        - "lương của cán bộ" 
        → rewrite: "chế độ tiền lương viên chức giảng viên người lao động"
        HƯỚNG DẪN:
        1. KHÔI PHỤC NGỮ CẢNH: Thay thế các đại từ (nó, trường này, ngành đó), câu rút gọn, hoặc chủ ngữ bị khuyết bằng các DANH TỪ RIÊNG cụ thể (tên trường, tên cơ sở, tên ngành, phương thức) ĐÃ XUẤT HIỆN trong hội thoại trước đó.
        ĐẶC BIỆT: Các cụm "dẫn chứng này", "thông tin trên", "quy định đó", "điều vừa nói", 
        "câu trả lời trước" → BẮT BUỘC tra lại lịch sử để tìm CHỦ ĐỀ CỤ THỂ đang được nhắc đến 
        rồi thay thế vào.
        
        Ví dụ:
        - Lịch sử: Bot vừa trả lời "kế hoạch nhiệm vụ năm học lưu trữ 20 năm theo quy định Bộ GDĐT"
        - User hỏi: "dẫn chứng này nằm ở quy định nào, điều khoản nào"
        - Rewrite thành: "Quy định về thời hạn lưu trữ kế hoạch nhiệm vụ năm học 20 năm 
                            nằm ở văn bản pháp lý nào, điều khoản nào?"
        2. KHÔNG BỊA ĐẶT: TUYỆT ĐỐI KHÔNG thêm thông tin hoàn toàn mới chưa từng được nhắc đến. KHÔNG thay đổi mục tiêu câu hỏi.
        3. GIỮ NGUYÊN NẾU ĐÃ RÕ RÀNG: Nếu câu hỏi mới nhất đã tự mang đủ ngữ nghĩa độc lập, hãy trả về NGUYÊN VĂN.
        4. KẾT QUẢ ĐẦU RA: Chỉ in ra ĐÚNG 1 CÂU truy vấn tiếng Việt, TUYỆT ĐỐI KHÔNG giải thích, KHÔNG có dấu ngoặc kép bọc ngoài.
        6. Viết lại câu hỏi mới nhất thành một câu truy vấn ĐỘC LẬP, ĐẦY ĐỦ NGỮ NGHĨA để máy tìm kiếm (Vector Search) có thể hiểu được chính xác mà không cần đọc lại lịch sử.
            + TRƯỜNG HỢP 1 (Dữ liệu đặc thù): NẾU câu hỏi liên quan đến CÁC THÔNG SỐ CỤ THỂ CỦA NGÀNH HỌC (Ví dụ: Điểm chuẩn, Điểm trúng tuyển, Chỉ tiêu, Tên ngành, Mã ngành, Tổ hợp môn). -> BẮT BUỘC thêm cụm "tại Phân hiệu TP.HCM (Mã tuyển sinh GSA)" vào cuối câu hỏi.
            + TRƯỜNG HỢP 2 (Dữ liệu dùng chung): NẾU câu hỏi liên quan đến QUY TRÌNH & THỦ TỤC (Ví dụ: Hồ sơ, Giấy tờ, Các bước nộp, Lệ phí, Tiêu chuẩn cộng điểm, Học phí, Ký túc xá, Quy chế). -> BẮT BUỘC GIỮ NGUYÊN câu hỏi gốc (TUYỆT ĐỐI KHÔNG thêm GSA hay Phân hiệu).
            + TRƯỜNG HỢP NGOẠI LỆ: Nếu câu hỏi quá ngắn hoặc không rõ thuộc trường hợp nào, hãy GIỮ NGUYÊN câu hỏi gốc để đảm bảo an toàn.
        7. MẶC ĐỊNH HỆ ĐÀO TẠO: Nếu người dùng hỏi chung chung về tuyển sinh, điểm chuẩn, tiêu chí, xét học bạ... mà KHÔNG nhắc đến hệ đào tạo nào, BẮT BUỘC bổ sung cụm từ "Hệ Đại học Chính quy" vào câu truy vấn.
        8. DỊCH THUẬT NGỮ TUYỂN SINH (Rất quan trọng):
            - Nhắc đến "học bạ" hoặc "kết quả học tập" -> BẮT BUỘC chèn thêm "Phương thức 2 (PT2)".
            - Nhắc đến "đánh giá năng lực" hoặc "ĐGNL" -> BẮT BUỘC chèn thêm "Phương thức 3 (PT3)".
        9. QUY TẮC BỔ SUNG THỜI GIAN CHO DỮ LIỆU TUYỂN SINH:
            NẾU câu hỏi của người dùng liên quan đến các chủ đề mang tính thời sự thay đổi theo từng năm (như: "chỉ tiêu", "điểm chuẩn", "học phí", "phương thức xét tuyển", "thông tin tuyển sinh"):

            Trường hợp 1: Nếu người dùng KHÔNG nhắc đến một năm cụ thể nào (ví dụ: 2024, 2025...), bạn BẮT BUỘC phải tự động chỉ duy nhất chèn thêm các từ khóa: "mới nhất", "dự kiến", "năm nay" vào câu truy vấn.

            Trường hợp 2: Nếu người dùng CÓ nhắc đến một năm cụ thể (ví dụ: "chỉ tiêu năm 2024"), hãy giữ nguyên mốc thời gian đó và tuyệt đối không thêm chữ "mới nhất".
        """
        # assume async predict exists
        enriched = await self.control_llm.ainvoke(prompt)
        print("==== RAW RESPONSE ====")
        print(user_message)
        print("======================")
        
        # fallback: if empty use original
        enriched_lines = (
            self._message_text(enriched).strip().splitlines() if enriched else []
        )
        enriched_txt = enriched_lines[0] if enriched_lines else user_message
        print("==== ENRICH QUERY ====")
        print(enriched_txt)
        print("======================")
        return enriched_txt

    # ---------------------------
    # LLM relevance check: ensure enriched_query actually matches the training QA
    # ---------------------------
    async def llm_relevance_check(
        self, enriched_query: str, matched_question: str, answer: str
    ) -> bool:
        prompt = f"""
        Bạn là chuyên gia đánh giá giữa câu hỏi tìm kiếm, câu hỏi trong cơ sở dữ liệu và câu trả lời, đánh giá độ phù hợp cho 1 hệ thống chat RAG tuyển sinh của trường {self.university_name}.

        Câu hỏi tìm kiếm (đã chuẩn hóa): "{enriched_query}"
        Câu hỏi DB: "{matched_question}"
        Câu trả lời chính thức: "{answer}"
        Nhiệm vụ:
        Xác định liệu "Câu hỏi DB + Câu trả lời" có thực sự trả lời đúng "Câu hỏi tìm kiếm" hay không.
        1. Hãy trả lời duy nhất chỉ một từ "true" khi:
        - Câu trả lời trực tiếp giải quyết đúng ý định (intent) của câu hỏi.
        - Câu hỏi mang tính chất chào hỏi (xin chào, hello, hi, bot ơi).
        2. Hãy trả lời duy nhất chỉ một từ "false" khi:
        - Chỉ trùng từ khóa nhưng khác ý nghĩa hoặc khác chủ đề
        - Câu trả lời quá chung chung, không giải quyết đúng câu hỏi
        - Câu hỏi người dùng hỏi X nhưng câu trả lời DB nói về Y
        - Câu hỏi người dùng quá rộng, câu trả lời DB quá cụ thể hoặc ngược lại
        3. Đặc biệt:
        - Nếu câu hỏi tìm kiếm là lời chào (xin chào, hello, hi): trả về true
        ---

        Chỉ trả về duy nhất một từ:
        "true" hoặc "false"
        """
        res = await self.control_llm.ainvoke(prompt)
        content = self._message_text(res)
        if not content:
            return False
        r = content.strip().lower()
        return (
            ("đúng" in r)
            or ("true" in r)
            or (r.startswith("đúng"))
            or (r.startswith("true"))
        )
    async def llm_listing_check(self, query: str) -> bool:
        prompt = f"""
        Bạn là chuyên gia phân loại câu hỏi cho hệ thống chatbot tuyển sinh của trường {self.university_name}.

        Câu hỏi người dùng: "{query}"

        Nhiệm vụ:
        Xác định liệu câu hỏi trên có yêu cầu một DANH SÁCH ĐẦY ĐỦ hay không.

        Trả về "true" nếu câu hỏi yêu cầu liệt kê toàn bộ, ví dụ:
        - Danh sách ngành đào tạo ("trường có những ngành gì", "liệt kê các ngành", "có bao nhiêu ngành")
        - Chỉ tiêu từng ngành ("chỉ tiêu các ngành năm nay", "mỗi ngành lấy bao nhiêu người")
        - Điểm chuẩn tất cả các ngành ("điểm chuẩn các ngành", "năm ngoái ngành nào lấy bao nhiêu điểm")
        - Học phí từng ngành / toàn bộ chương trình ("học phí các ngành", "bảng học phí")
        - Các phương thức xét tuyển ("có những phương thức xét tuyển nào")
        - Danh sách tổ hợp môn ("các tổ hợp môn xét tuyển", "tổ hợp nào được dùng")
        - Danh sách học bổng / chính sách hỗ trợ ("có những loại học bổng nào")
        - Danh sách cơ sở / campus ("trường có mấy cơ sở", "các cơ sở đào tạo")
        - Bất kỳ câu hỏi nào dùng từ: "tất cả", "toàn bộ", "các ... là gì", "liệt kê", "danh sách", "có những ... nào", "bao nhiêu ngành/chuyên ngành"

        Trả về "false" nếu câu hỏi chỉ hỏi về MỘT ngành / MỘT thông tin cụ thể, ví dụ:
        - "ngành CNTT điểm chuẩn bao nhiêu"
        - "học phí ngành Kinh tế là bao nhiêu"
        - "xét tuyển học bạ cần điều kiện gì"
        - "trường có ngành Y không"
        - câu hỏi chung chung không cần liệt kê

        Chỉ trả về duy nhất một từ:
        "true" nếu cần danh sách đầy đủ, "false" nếu không cần.
        """
        res = await self.control_llm.ainvoke(prompt)
        content = self._message_text(res)
        if not content:
            return False
        r = content.strip().lower()
        return (
            ("đúng" in r)
            or ("true" in r)
            or r.startswith("đúng")
            or r.startswith("true")
        )
    
    async def llm_admission_check(self, query: str) -> bool:
        prompt = f"""
        Bạn là chuyên gia phân loại câu hỏi cho hệ thống chatbot tuyển sinh của trường {self.university_name}.

        Câu hỏi người dùng: "{query}"

        Nhiệm vụ:
        Xác định liệu câu hỏi trên có thuộc lĩnh vực tuyển sinh đại học

        Phạm vi tuyển sinh bao gồm:
        - Xét tuyển, điểm chuẩn, phương thức xét tuyển (học bạ, ĐGNL, thi THPT...)
        - Ngành đào tạo, chuyên ngành, tổ hợp môn xét tuyển
        - Học phí
        - Điều kiện xét tuyển, hồ sơ đăng ký, quy trình nộp hồ sơ
        - Chỉ tiêu tuyển sinh, thời gian tuyển sinh
        

        Chỉ trả về duy nhất một từ:
        "true" nếu thuộc tuyển sinh, "false" nếu không thuộc.
        """
        res = await self.control_llm.ainvoke(prompt)
        content = self._message_text(res)
        if not content:
            return False
        r = content.strip().lower()
        return (
            ("đúng" in r)
            or ("true" in r)
            or r.startswith("đúng")
            or r.startswith("true")
        )

    async def llm_document_recommendation_check(
        self, enriched_query: str, context: str
    ) -> bool:

        prompt = f"""Bạn là hệ thống phân loại dữ liệu (Pre-check) cho chatbot RAG tư vấn tuyển sinh của trường {self.university_name}.

        Nhiệm vụ của bạn là kiểm tra xem đoạn tài liệu (Context) được trích xuất từ cơ sở dữ liệu CÓ GIÁ TRỊ THAM KHẢO để trả lời câu hỏi của người dùng hay không.

        === LUẬT PHÂN LOẠI ===
        Chỉ trả về duy nhất 1 từ: "document" hoặc "nope". Không giải thích gì thêm.

        - Trả về "document" nếu Context thỏa mãn ÍT NHẤT MỘT trong các điều kiện sau:
            1. Chứa câu trả lời trực tiếp và chính xác cho câu hỏi.
            2. Chứa thông tin LIÊN QUAN, CÓ ÍCH để dùng làm tài liệu tham khảo. (Ví dụ: Người dùng hỏi về Phân hiệu nhưng context có số liệu của Toàn trường; Người dùng hỏi định mức giờ chuẩn cụ thể nhưng context có chính sách/thông tư quy định chung về giờ chuẩn).
            3. Chứa thông tin để chatbot có thể giải thích, hướng dẫn hoặc dẫn dắt người dùng đến đúng vấn đề.

        - Trả về "nope" CHỈ KHI Context rơi vào các trường hợp sau:
            1. HOÀN TOÀN LẠC ĐỀ: Đọc "Nội dung Document Base (context)" thấy không có bất kỳ sự liên quan nào đến ý định của người dùng.
            2. TRÙNG TỪ KHÓA NHƯNG KHÁC Ý NGHĨA: (Ví dụ: Hỏi "Điểm chuẩn" nhưng context nói về "Tiêu chuẩn phòng cháy chữa cháy").

        === DỮ LIỆU ĐẦU VÀO ===
        Câu hỏi người dùng: "{enriched_query}"

        Nội dung Document Base (context):
        \"\"\"
        {context}
        \"\"\"
        """

        res = await self.control_llm.ainvoke(prompt)
        r = self._message_text(res).strip().lower()
        if r not in ["document", "recommendation", "nope"]:
            r = "nope"
        return r

    async def llm_suitable_for_recommedation_check(
        self, enriched_query: str, context: str
    ) -> bool:
        prompt = f"""
        Bạn là hệ thống kiểm tra mức độ liên quan giữa câu hỏi người dùng có liên quan đến các nội dung tư vấn ngành học hay tư vấn cho cá nhân dựa theo hồ sơ của học sinh hoặc những câu liên quan đến RIASEC, học bạ, GPA, sở thích, nguyện vọng cá nhân; hoặc yêu cầu so sánh ngành theo profile; hoặc yêu cầu gợi ý ngành phù hợp cho chatbot RAG tư vấn tuyển sinh của trường {self.university_name}.

        Yêu cầu:
        - Chỉ trả về "true" nếu câu hỏi có liên quan đến các nội dung đó.
        - Trả về "false" nếu câu hỏi không liên quan đến các nội dung đó.

        Câu hỏi người dùng: "{enriched_query}"

        
        Hãy TRẢ LỜI DUY NHẤT:
        - "true" → nếu câu hỏi có liên quan đến các nội dung đó 
        - "false" → nếu câu hỏi không liên quan đến các nội dung đó
        """

        res = await self.control_llm.ainvoke(prompt)
        content = self._message_text(res)
        if not content:
            return False
        r = content.strip().lower()
        return (
            ("đúng" in r)
            or ("true" in r)
            or (r.startswith("đúng"))
            or (r.startswith("true"))
        )

    async def response_from_riasec_result(
        self, riasec_result: schemas.RiasecResultCreate
    ):
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
            res = await self.control_llm.ainvoke(prompt)
            return self._message_text(res).strip()

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
                    {"input": last_user_msg or ""}, {"output": inter.message_text}
                )
                last_user_msg = None

        # Nếu cuối cùng là tin nhắn user chưa được phản hồi
        if last_user_msg:
            memory.save_context({"input": last_user_msg}, {"output": ""})

    def update_faq_statistics(self, db: Session, response_id: int, intent_id: int = 1):

        try:
            response = (
                db.query(ChatInteraction)
                .filter(
                    ChatInteraction.interaction_id == response_id,
                    ChatInteraction.is_from_bot == True,
                )
                .first()
            )

            if not response:
                raise ValueError("Chatbot response not found")

            faq = FaqStatistics(response_from_chat_id=response_id, intent_id=intent_id)
            db.add(faq)
            db.commit()

        except Exception as e:
            db.rollback()
            print(f"Error updating FaqStatistics: {e}")

    async def stream_response_from_context(
        self,
        query: str,
        context: str,
        session_id: int,
        user_id: int,
        intent_id: int,
        message: str,
        query_embedding: Optional[List[float]] = None,
        current_audience_id: int = None,
        current_intent_id: int = None,
        confidence: float = 5.0,
    ):
        print("vào doc stream")
        
        db = SessionLocal()
        suggestion_threshold = float(os.getenv("CONFIDENCE_SCORE", 0.35))
        try:
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id,
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
                    session_id=session_id,
                )

                db.add(user_msg)
                db.flush()
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")
            full_response = ""
            suggestion = None

            print("→ going to LLM context")
            print(context)
            prompt = f"""Bạn là một chatbot tra cứu thông tin chuyên nghiệp của trường {self.university_name}
            Hãy coi mọi thông tin được cung cấp trong Context chính là "kiến thức của nhà trường" và "kiến thức của bạn".
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === THÔNG TIN THAM KHẢO ===
            {context}
            === CÂU HỎI ===
            {query}
            === PHONG CÁCH TRẢ LỜI ===
            Cách trả lời:
                - ĐỐI VỚI DỮ LIỆU SỐ LƯỢNG/CHỈ TIÊU: BẮT BUỘC trình bày dưới dạng danh sách gạch đầu dòng (bullet points) thật gọn gàng, dễ nhìn.
                - TUYỆT ĐỐI KHÔNG in ra định dạng bảng chứa các ký tự "|".
                - Dễ hiểu
                - Thân thiện
                - Trả lời bằng tiếng Việt
                - Dùng ngôn ngữ đời thường
                - Dùng Markdown linh hoạt: chỉ dùng tiêu đề ## và gạch đầu dòng khi câu trả lời có nhiều mục rõ ràng. Câu trả lời ngắn thì viết thành đoạn văn tự nhiên, không cần chia heading.
                - Nếu trong câu trả lời có đường dẫn thì hãy markdown đường dẫn
            Không được:
                - Lặp lại ý người dùng
                - Dùng ngôn ngữ AI máy móc, robot
                - TUYỆT ĐỐI KHÔNG ĐƯỢC LẤP LIẾM, TỰ GÁN THÔNG TIN
            Cách phản hồi:
                - Trả lời trực tiếp câu hỏi
                - Nếu cần, hướng dẫn từng bước
                - Gợi ý thông tin liên quan hữu ích
                - Nếu có đường dẫn liên quan đến nội dung người dùng muốn biết, có thể gợi ý để họ tự tìm hiểu thêm, TUYỆT ĐỐI không được lấy đường dẫn không liên quan đến nội dung trả lời
            === Thông tin các phòng ban bạn có thể tham khảo để gợi ý nếu cần thiết tùy tình hướng ===
                Phòng Tổ chức Hành chính
                Tham mưu về tổ chức cán bộ, tuyển dụng, quy hoạch, bồi dưỡng và chế độ chính sách cho viên chức.
                Làm công tác hành chính tổng hợp, văn thư, lưu trữ.
                Quản lý công tác bảo vệ, an ninh trật tự.
                Phụ trách công tác y tế ban đầu.
                Quản lý nhà khách giáo viên, phòng họp và phòng làm việc của Ban Giám đốc.
                Địa chỉ: Phòng 1, Phòng 2-3 Nhà D3, số 450 Đường Lê Văn Việt, P. Tăng Nhơn Phú, TP. Hồ Chí Minh
                Điện thoại: 028 38966798
                Email: tochuchanhchinh@utc2.edu.vn

                Đào tạo
                Tổ chức và quản lý đào tạo các hệ.
                Xếp thời khóa biểu, xếp lịch thi, rà soát kế hoạch học tập năm học.
                Quản lý kết quả học tập, xét học bổng, cảnh báo học vụ, tốt nghiệp.
                Cấp bảng điểm, xử lý các quyết định liên quan đến sinh viên như chuyển điểm, thôi học, chuyển hệ.
                Phụ trách công tác tốt nghiệp, xét nhận đề tài, thành lập hội đồng.
                Tổ chức đào tạo GDQP-AN.
                Địa chỉ: Phòng 8, 9, 10 Nhà D3, số 450 Lê Văn Việt, P. Tăng Nhơn Phú A, TP. Hồ Chí Minh
                Điện thoại: (028) 3896 2018; (028) 3730 7908
                Email: bandaotao@utc2.edu.vn

                Phòng Khảo thí và Đảm bảo chất lượng đào tạo
                Tổ chức các kỳ thi kết thúc học phần.
                Quản lý đề thi, chấm thi, và kết quả thi.
                Làm công tác đảm bảo chất lượng đào tạo.
                Khảo sát ý kiến người học và các bên liên quan.
                Phụ trách kiểm định, đánh giá ngoài và tuyển sinh.
                Địa chỉ: Phòng 108, 109, 110 Nhà E10, số 451 Lê Văn Việt, P. Tăng Nhơn Phú A, TP. Hồ Chí Minh
                Điện thoại: 028 3896 2819; 028 3730 6883; 028 3730 7120
                Email: bankt-dbcl@utc2.edu.vn
                
                Công tác chính trị và Sinh viên
                Phụ trách công tác chính trị, tư tưởng, tuyên truyền.
                Quản lý hồ sơ sinh viên.
                Tổ chức các hoạt động văn hóa, thể thao, thi đua khen thưởng.
                Hỗ trợ sinh viên về chính sách, học bổng, vay vốn tín dụng.
                Quản lý thẻ sinh viên, cố vấn học tập, đánh giá rèn luyện.
                Xét và đề nghị kỷ luật sinh viên khi vi phạm.
                Địa chỉ: Phòng 6, Phòng 15 Nhà D3, số 450 Lê Văn Việt, P. Tăng Nhơn Phú A, TP. Hồ Chí Minh
                Điện thoại: 028 3736 0564
                Email: phongctctsv@utc2.edu.vn
                
                Khoa học Công nghệ & Đối ngoại
                Xây dựng và triển khai kế hoạch khoa học công nghệ.
                Quản lý, theo dõi các hoạt động nghiên cứu khoa học.
                Phụ trách công tác đối ngoại.
                Hỗ trợ các hoạt động truyền thông, bồi dưỡng ngắn hạn, giảng bài theo phân công.
                Địa chỉ: Nhà D7, số 450 Lê Văn Việt, P. Tăng Nhơn Phú A, TP. Hồ Chí Minh
                Điện thoại: 028 3736 1575
                Email: stic@utc2.edu.vn
                
                Phòng Thiết bị - Quản trị
                Quản lý cơ sở vật chất, trang thiết bị, công trình phục vụ hoạt động của trường.
                Phụ trách công tác quản trị, sửa chữa, bảo trì và vận hành các hạng mục của nhà trường.
                Địa chỉ: Nhà D7, số 450 Lê Văn Việt, P. Tăng Nhơn Phú, TP. Hồ Chí Minh
                Email: thietbiquantri@utc2.edu.vn
                Trong dữ liệu bạn gửi chưa thấy ghi rõ số điện thoại của phòng này.
                
                Phòng Tài chính - Kế toán
                Quản lý, phân phối và giám sát việc sử dụng kinh phí của trường.
                Thu học phí.
                Chi trả học bổng và trợ cấp xã hội cho sinh viên.
                Địa chỉ: Phòng 4, 5, 7 Nhà D3, số 450 Lê Văn Việt, P. Tăng Nhơn Phú, TP. Hồ Chí Minh
                Điện thoại: (028) 3896 2174
                Email: taichinhketoan@utc2.edu.vn
                
                Ban Thanh tra
                Kiểm tra việc thực hiện giờ giấc lên lớp của giảng viên.
                Giám sát các kỳ thi tuyển sinh, thi kết thúc học phần, thi sát hạch ngoại ngữ.
                Tiếp công dân, xử lý đơn thư phản ánh, kiến nghị, khiếu nại, tố cáo.
                Kiểm tra sau tuyển sinh và các nội dung liên quan đến tính minh bạch trong đào tạo.
                Địa chỉ: Nhà E1, khu Giảng đường, số 451 Lê Văn Việt, P. Tăng Nhơn Phú A, TP. Hồ Chí Minh
                Điện thoại: 028 3730 9469
                Email: thanhtra@utc2.edu.vn

                Trung tâm Đào tạo thực hành và Chuyển giao Công nghệ GTVT
                Quản lý, khai thác phòng thực hành và thí nghiệm.
                Tổ chức đào tạo các phần mềm chuyên ngành, bồi dưỡng kiến thức trong và ngoài trường.
                Thực hiện dịch vụ khoa học công nghệ, lao động sản xuất, nghiên cứu và chuyển giao công nghệ.
                Địa chỉ: Nhà E7, số 451 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Hồ Chí Minh
                Điện thoại: (028) 3736 0512
                Fax: (028) 3736 0676
                Email: daotaothuchanh@utc2.edu.vn
                Website: http://dept.utc2.edu.vn/trungtamdaotao/

                Trung tâm Thông tin - Thư viện
                Quản lý hệ thống CNTT toàn trường.
                Tổ chức thu thập, khai thác tài liệu phục vụ học tập và giảng dạy.
                Cung cấp dịch vụ thư viện, mượn trả sách và tài liệu cho sinh viên, giảng viên.
                Địa chỉ: Nhà C3, số 451 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Hồ Chí Minh
                Điện thoại: 028 3730 9492
                Website: http://thuvien.utc2.edu.vn/
                Email: thongtinthuvien@utc2.edu.
                
                Ban Quản lý Ký túc xá tại UTC2
                Quản lý khu ký túc xá.
                Tổ chức tiếp nhận, bố trí chỗ ở cho sinh viên nội trú.
                Ký, thực hiện và đình chỉ hợp đồng nội trú theo quy định.
                Xử lý các trường hợp vi phạm nội quy ký túc xá.
                Địa chỉ: Số 450 Lê Văn Việt, Phường Tăng Nhơn Phú, TP. Hồ Chí Minh
                Điện thoại: (028) 3730.9099
                Email: kytucxa@utc2.edu.vn
                Fanpage: https://www.facebook.com/ktxutc2
            === KỈ LUẬT THÉP ===
            1. BẢO VỆ TÍNH CHÍNH XÁC CỦA ĐƠN VỊ VÀ LIÊN HỆ (CẤM RÂU ÔNG NỌ CẮM CẰM BÀ KIA):
                - Khi bạn khuyên người dùng liên hệ một đơn vị/phòng ban cụ thể (Ví dụ: Phòng Đào tạo, Phòng Khảo thí), BẮT BUỘC chỉ được cung cấp thông tin liên hệ (SĐT, Email) CỦA CHÍNH XÁC PHÒNG BAN ĐÓ (phải khớp tên).
                - LỆNH CẤM THAY THẾ: TUYỆT ĐỐI KHÔNG copy số điện thoại của các phòng ban khác (như Tuyển sinh, Tổ chức hành chính, CTCTSV...) dán vào để "chữa cháy" hoặc "gợi ý thêm" khi không tìm thấy số của phòng ban cần thiết.
                - CÁCH XỬ LÝ KHI THIẾU DATA: Nếu khuyên liên hệ "Phòng X" nhưng trong context KHÔNG CÓ số điện thoại của "Phòng X", BẮT BUỘC chỉ dừng lại ở lời khuyên và nói rõ: "Hiện tại hệ thống chưa có thông tin liên hệ trực tiếp của phòng này, bạn vui lòng tra cứu trên website trường nhé." (CẤM tự động liệt kê số của Tổ chức hành chính/Đường dây nóng ra để bù đắp).
            2. - QUY TẮC TRẢ LỜI HỌC PHÍ (CHỈ ÁP DỤNG CHO ĐẠI HỌC):
                + Bạn là chatbot của Trường ĐẠI HỌC Giao thông Vận tải. TUYỆT ĐỐI KHÔNG trích xuất hoặc liệt kê học phí của các cấp học không liên quan bao gồm: Mầm non, Tiểu học, Trung học cơ sở (THCS), Trung học phổ thông (THPT).
                + Nếu Context kéo lên dữ liệu của các cấp học phổ thông này, hãy thẳng tay BỎ QUA, chỉ giữ lại dữ liệu học phí bậc Đại học (hoặc Sau đại học nếu có). Việc bỏ qua này KHÔNG BỊ TÍNH là lỗi tóm tắt thiếu dữ liệu.
            3. - QUY TẮC TRẢ LỜI VỀ HỌC PHÍ KHI THIẾU DỮ LIỆU:
                + Khi người dùng hỏi về "Học phí" của trường/ngành, BẮT BUỘC phải tìm số liệu học phí CỤ THỂ của trường Đại học GTVT (UTC2) (thường tính bằng VNĐ/tín chỉ hoặc VNĐ/tháng/ngành).
                + NẾU TRONG TÀI LIỆU KHÔNG CÓ con số cụ thể này (hoặc chỉ có Nghị định quy định mức trần chung của Chính phủ), bạn BẮT BUỘC phải trả lời: "Dạ, hiện tại hệ thống chưa cập nhật thông tin học phí chính thức năm 2025-2026 của Phân hiệu UTC2. Bạn vui lòng theo dõi thêm thông báo trên website/fanpage của trường hoặc liên hệ trực tiếp bộ phận Tư vấn tuyển sinh nhé."
                + LỆNH CẤM: TUYỆT ĐỐI KHÔNG trích xuất các con số "Mức trần học phí" từ Nghị định của Chính phủ để trả lời thay thế, vì điều này sẽ gây hiểu lầm nghiêm trọng cho sinh viên.
            4. - PHÂN BIỆT GIỮA "SỐ LIỆU THỰC TẾ" VÀ "CÔNG THỨC/QUY ĐỊNH":
                + Khi người dùng hỏi cụ thể về "số liệu", "tỷ lệ %", "điểm số", "bao nhiêu", bạn BẮT BUỘC phải tìm các CON SỐ THỐNG KÊ THỰC TẾ.
                + LỆNH CẤM: TUYỆT ĐỐI KHÔNG được trả lời "Có số liệu" nhưng sau đó lại đi liệt kê "Công thức tính toán" hoặc "Định nghĩa khái niệm" từ các Thông tư, Nghị định.
                + Xử lý khi thiếu con số: Nếu tài liệu chỉ dạy "cách tính" mà không có con số thực tế, BẮT BUỘC trả lời trung thực: "Dạ, hiện tại hệ thống chưa cập nhật con số thống kê thực tế về [chủ đề] của Phân hiệu UTC2. Tài liệu hiện tại chỉ có quy định về cách tính..."
            5. - LOGIC ĐỒNG NHẤT CHO CÂU HỎI CÓ/KHÔNG (YES/NO QUESTIONS):
                + Khi người dùng hỏi các câu mang tính xác nhận hành vi (Ví dụ: Có được không? Có cho không? Được phép không?), từ khẳng định/phủ định ở ĐẦU CÂU trả lời BẮT BUỘC phải đồng nhất tuyệt đối về mặt ngữ nghĩa với phần giải thích phía sau.
                + Nếu nội dung quy định là cấm/không được phép, BẮT BUỘC mở đầu bằng: "Dạ KHÔNG," hoặc "Không được phép,". 
                + LỆNH CẤM: TUYỆT ĐỐI KHÔNG dùng từ "Có" ở đầu câu như một từ đệm giao tiếp nếu nội dung thực tế phía sau mang ý nghĩa phủ định (Ví dụ: Cấm trả lời kiểu "Có. Theo nội quy là không được...").
                - KỸ NĂNG XỬ LÝ NGOẠI LỆ (EXCEPTION HANDLING): Khi người dùng hỏi về một "Đối tượng/Tình huống cụ thể", BẮT BUỘC kiểm tra xem nó chịu sự chi phối của "Quy định chung" hay nằm trong "Trường hợp ngoại lệ" của tài liệu.
                    CÁCH TRẢ LỜI: Nếu đối tượng là NGOẠI LỆ, phải đưa ra câu trả lời trực tiếp cho tính chất của ngoại lệ đó ở ngay đầu câu (Ví dụ: "Dạ ĐƯỢC PHÉP ạ, vì..."). TUYỆT ĐỐI KHÔNG dùng cấu trúc "Dạ KHÔNG (theo luật chung)... ngoại trừ (luật riêng)..." gây mâu thuẫn logic và khó hiểu.
            6. - QUY TẮC KHÔNG BỎ SÓT KHI LIỆT KÊ:
                + Với BẤT KỲ câu hỏi nào (nhân sự, ngành học, phòng ban, chính sách...),
                nếu trong context có NHIỀU ĐỐI TƯỢNG thỏa mãn câu hỏi, BẮT BUỘC 
                liệt kê ĐẦY ĐỦ TẤT CẢ, không dừng lại ở kết quả đầu tiên tìm được.
                + Trước khi trả lời, BẮT BUỘC tự hỏi: "Trong context còn đối tượng nào 
                khác thỏa mãn câu hỏi này không?" — nếu CÓ thì phải đưa vào câu trả lời.
                + LỆNH CẤM: TUYỆT ĐỐI KHÔNG dừng lại ở kết quả đầu tiên chỉ vì "nghe 
                có vẻ đủ rồi". Thiếu sót trong liệt kê là sai, dù chỉ bỏ một đối tượng.
            7. - QUY TẮC PHÂN BIỆT NGUỒN KHI CONTEXT CÓ NHIỀU ĐỐI TƯỢNG CÙNG LOẠI:
                + Khi câu hỏi hỏi về một ĐỐI TƯỢNG CỤ THỂ được nêu tên (VD: một khoa,
                một phòng ban, một người, một ngành...), BẮT BUỘC chỉ sử dụng thông 
                tin từ chunk có TÊN ĐỐI TƯỢNG KHỚP CHÍNH XÁC với câu hỏi.
                + Nếu context chứa nhiều chunk từ nhiều đối tượng khác nhau cùng loại
                (VD: nhiều khoa, nhiều phòng ban), TUYỆT ĐỐI KHÔNG lấy thông tin từ 
                chunk của đối tượng không được hỏi đến, dù cấu trúc hay chức danh 
                có giống nhau.
                + Trước khi trả lời, BẮT BUỘC tự kiểm tra: "Chunk này có nhắc đến tên 
                đối tượng được hỏi không?" — Nếu KHÔNG → bỏ qua chunk đó.
            8. - QUY TẮC PHÂN BIỆT CẤP CHỨC VỤ:
                + "Trưởng/Phó Trưởng KHOA" ≠ "Trưởng/Phó Trưởng BỘ MÔN".
                + TUYỆT ĐỐI KHÔNG nhầm lẫn hoặc gộp 2 cấp này vào chung một danh 
                sách lãnh đạo khoa. Chỉ liệt kê đúng cấp được hỏi.
            === HƯỚNG DẪN XỬ LÝ LƯU Ý ===
            - Dựa vào thông tin tham khảo trên được cung cấp
            - Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
            - Bạn là chatbot tra cứu thông tin chuyên nghiệp của {self.university_name}, nếu câu hỏi yêu cầu thông tin của một trường khác hay phân hiệu khác thì nói rõ là không có dữ liệu trong hệ thống hiện tại
            - Nếu không tìm thấy thông tin → Nói rõ hệ thống chưa có dữ liệu, →  Có thể chọn đường dẫn chọn phù hợp từ context để gợi ý
            chỉ khi đường dẫn đó TRỰC TIẾP xử lý đúng vấn đề được hỏi.
            - KHI NGƯỜI DÙNG YÊU CẦU LIỆT KÊ SỐ LƯỢNG: Bạn bắt buộc phải rà soát toàn bộ bảng trong tài liệu và liệt kê ĐẦY ĐỦ TẤT CẢ các ngành có trong ngữ cảnh. TUYỆT ĐỐI KHÔNG ĐƯỢC BỎ SÓT, KHÔNG ĐƯỢC TỰ Ý TÓM TẮT BỚT NGÀNH. Hãy đọc từ trên xuống dưới một cách cẩn thận.
            - Khi trả lời về một đơn vị/phòng ban cụ thể, CHỈ sử dụng thông tin
                từ chunk có heading KHỚP CHÍNH XÁC với tên đơn vị được hỏi.
                KHÔNG lấy thông tin (website, email, SĐT) từ chunk của đơn vị khác
                dù tên có vẻ tương tự.
            - Nếu câu hỏi chỉ là chào hỏi, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
            - Năm của dữ liệu: lấy từ heading trong context (VD: "Đề án tuyển sinh 2026").
    KHÔNG tự suy đoán hoặc copy năm từ câu hỏi của người dùng nếu context không xác nhận.
            - Cuối câu trả lời, nếu phù hợp, hãy gợi ý một chủ đề liên quan đến context
                mà người dùng có thể quan tâm tiếp theo (điểm chuẩn, học bổng, 
                chuyên ngành, học phí...). Thay đổi gợi ý theo ngữ cảnh câu hỏi, 
                không lặp lại cùng một câu mẫu.
            """
            full_response = ""
            async for chunk in self.answer_llm.astream(prompt):
                text = self._message_text(chunk)
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop
            print(full_response)
            memory.save_context({"input": message}, {"output": full_response})

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id,
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            self.update_faq_statistics(db, bot_msg.interaction_id, intent_id=intent_id)
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close()

    async def stream_response_from_context_tuyensinh(
        self,
        query: str,
        context: str,
        session_id: int,
        user_id: int,
        intent_id: int,
        message: str,
        query_embedding: Optional[List[float]] = None,
        current_audience_id: int = None,
        current_intent_id: int = None,
        confidence: float = 5.0,
    ):
        print("vào doc stream")
        print(context)
        db = SessionLocal()
        suggestion_threshold = float(os.getenv("CONFIDENCE_SCORE", 0.35))
        try:
            if not user_id:
                # 🧩 1. Lưu tin nhắn người dùng
                user_msg = ChatInteraction(
                    message_text=message,
                    timestamp=datetime.now(),
                    rating=None,
                    is_from_bot=False,
                    sender_id=None,
                    session_id=session_id,
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
                    session_id=session_id,
                )

                db.add(user_msg)
                db.flush()
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")
            full_response = ""
            suggestion = None

            print("→ going to LLM context tuyensinh")

            prompt = f"""Bạn là một chatbot tra cứu thông tuyển sinh chuyên nghiệp của trường {self.university_name}, mã tuyển sinh GSA
            Hãy coi mọi thông tin được cung cấp trong Context chính là "kiến thức của nhà trường" và "kiến thức của bạn".
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === THÔNG TIN THAM KHẢO ===
            {context}
            === CÂU HỎI ===
            {query}
            === PHONG CÁCH TRẢ LỜI ===
            Cách trả lời:
                - ĐỐI VỚI DỮ LIỆU SỐ LƯỢNG/CHỈ TIÊU: BẮT BUỘC trình bày dưới dạng danh sách gạch đầu dòng (bullet points) thật gọn gàng, dễ nhìn.
                - TUYỆT ĐỐI KHÔNG in ra định dạng bảng chứa các ký tự "|".
                - Nếu có nhiều phần (Chính quy, Chất lượng cao, Từ xa...), hãy dùng tiêu đề (Heading 2 hoặc 3) để phân chia rõ ràng trước khi gạch đầu dòng.
                - Dễ hiểu
                - Thân thiện
                - Trả lời bằng tiếng Việt
                - Dùng ngôn ngữ đời thường
                - Dùng Markdown linh hoạt: chỉ dùng tiêu đề ## và gạch đầu dòng khi câu trả lời có nhiều mục rõ ràng. Câu trả lời ngắn thì viết thành đoạn văn tự nhiên, không cần chia heading.
                - Nếu trong câu trả lời có đường dẫn thì hãy markdown đường dẫn
            Không được:
                - Lặp lại ý người dùng
                - Dùng ngôn ngữ AI máy móc, robot
                - TUYỆT ĐỐI KHÔNG ĐƯỢC LẤP LIẾM, TỰ GÁN THÔNG TIN
            Cách phản hồi:
                - Trả lời trực tiếp câu hỏi
                - Nếu cần, hướng dẫn từng bước
                - Gợi ý thông tin liên quan hữu ích
                - Nếu có đường dẫn liên quan đến nội dung người dùng muốn biết, có thể gợi ý để họ tự tìm hiểu thêm, TUYỆT ĐỐI không được lấy đường dẫn không liên quan đến nội dung trả lời
                === TỪ ĐIỂN ĐỒNG NGHĨA (BẮT BUỘC GHI NHỚ) ===
                Trong mọi tài liệu và câu hỏi, bạn PHẢI TỰ ĐỘNG HIỂU các cụm từ sau đây là ĐỒNG NGHĨA và CHỈ CÙNG MỘT CƠ SỞ ĐÀO TẠO:
                1. "Phân hiệu Trường Đại học Giao thông Vận tải tại TP. Hồ Chí Minh"
                2. "Phân hiệu tại TP.HCM"
                3. "UTC2"
                4. "Mã trường GSA" / "Mã tuyển sinh GSA"
                (Ví dụ: Nếu tài liệu ghi là "Mã tuyển sinh GSA", bạn được quyền hiểu và trả lời cho câu hỏi về "UTC2").
            === KỶ LUẬT THÉP (BẮT BUỘC TUÂN THỦ TÙY TÌNH HUỐNG) ===
            1. CHỐNG BỊA ĐẶT ĐA HỆ ĐÀO TẠO: Ngành nào CÓ TÊN TRONG BẢNG CỦA HỆ NÀO thì mới được liệt kê vào hệ đào tạo đó. TUYỆT ĐỐI KHÔNG copy số liệu của hệ Chính quy xuống gán cho hệ khác. Nếu bảng của hệ đó không có tên ngành, BẮT BUỘC kết luận: "Tài liệu không có thông tin chỉ tiêu cho ngành này ở hệ [Tên hệ]."
            2. PHÂN BIỆT NHÓM NGÀNH VÀ NGÀNH: TUYỆT ĐỐI KHÔNG lấy tổng chỉ tiêu của cả một "Nhóm ngành" để gán cho một "Ngành" đơn lẻ. Nếu chỉ có số liệu nhóm, trả lời: "Tài liệu hiện tại chỉ thống kê chỉ tiêu tổng của cả Nhóm ngành [Tên nhóm] là [Số lượng], chưa có số liệu tách riêng cho ngành này."
            3. TÍNH CHÍNH XÁC CỦA ĐƠN VỊ: Khi trả lời về một đơn vị/phòng ban, CHỈ dùng thông tin từ chunk có tiêu đề khớp chính xác với đơn vị đó. Không lấy râu ông nọ cắm cằm bà kia.
            4. KIỂM CHỨNG HEADING (CHAIN-OF-THOUGHT BẮT BUỘC):
                - Bạn phải tự đối chiếu: Nếu [Tên Heading] thuộc hệ "Vừa làm vừa học" hoặc "Từ xa" nhưng câu hỏi của người dùng là "Chính quy" (hoặc không hỏi hệ nào), BẮT BUỘC kết luận: "Hiện tại tài liệu chỉ kéo lên quy định của hệ Vừa làm vừa học/Từ xa, chưa có thông tin tiêu chí cho hệ Chính quy."
                - TUYỆT ĐỐI KHÔNG được lấy số liệu của Heading này rồi tự ý đổi tên thành hệ đào tạo khác để trả lời.
            5. BẮT BUỘC KHAI BÁO HỆ ĐÀO TẠO/PHẠM VI: 
                - Khi trích xuất bất kỳ tiêu chí, điểm số, hay quy định nào từ ngữ cảnh, BẮT BUỘC phải ghi rõ quy định đó thuộc Hệ đào tạo nào (Chính quy, Vừa làm vừa học, Từ xa...) dựa vào Tiêu đề (Heading) của đoạn ngữ cảnh chứa thông tin đó.
                - TUYỆT ĐỐI KHÔNG trả lời chung chung kiểu "Tài liệu cho thấy tiêu chí là...". 
                - Mẫu trả lời đúng: "Đối với Hệ [Tên hệ đào tạo theo Heading], tài liệu quy định tiêu chí là..."
                - CẢNH BÁO TÌNH HUỐNG LỘN XỘN: Nếu người dùng hỏi Hệ Chính quy, nhưng context chỉ kéo lên quy định của Hệ Vừa làm vừa học, BẮT BUỘC phải nói: "Hiện tại tài liệu chưa có quy định chi tiết cho Hệ Chính quy. Tuy nhiên, đối với Hệ Vừa làm vừa học, quy định là..."
            6. SÀNG LỌC ĐÚNG ĐỐI TƯỢNG VÀ ĐIỀU KIỆN (CỰC KỲ QUAN TRỌNG):
                - BẮT BUỘC phải đọc kỹ các ĐIỀU KIỆN RÀNG BUỘC trong câu hỏi của người dùng (Ví dụ: Năm tốt nghiệp, Hệ đào tạo, Diện xét tuyển).
                - Khi trích xuất thông tin từ Context, CHỈ ĐƯỢC PHÉP giữ lại những quy định/thủ tục ÁP DỤNG ĐÚNG cho điều kiện của người dùng.
                - LỆNH CẤM: TUYỆT ĐỐI KHÔNG copy thừa thãi các quy định dành cho đối tượng khác. (Ví dụ: Nếu người dùng hỏi cách đăng ký cho thí sinh "tốt nghiệp năm 2026", CẤM liệt kê thủ tục dành riêng cho thí sinh "tốt nghiệp trước năm 2026", trừ khi thủ tục đó là quy định chung cho tất cả).
                - Hướng dẫn gom nhóm: Nếu có các trường hợp ngoại lệ (ví dụ: dùng minh chứng điểm cộng, xét tuyển thẳng), phải ghi rõ "CHỈ ÁP DỤNG NẾU bạn thuộc diện..." để người dùng không bị hiểu lầm.
            === HƯỚNG DẪN XỬ LÝ ===
            - Dựa vào thông tin tham khảo trên được cung cấp
            - Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
            - Bạn là chatbot tra cứu thông tin chuyên nghiệp của {self.university_name}, nếu câu hỏi yêu cầu thông tin của một trường khác hay phân hiệu khác thì nói rõ là không có dữ liệu trong hệ thống hiện tại
            - Nếu không tìm thấy thông tin → Nói rõ hệ thống mục tuyển sinh chưa có dữ liệu, →  Có thể chọn đường dẫn chọn phù hợp từ context để gợi ý
            chỉ khi đường dẫn đó TRỰC TIẾP xử lý đúng vấn đề được hỏi.
            - QUY TẮC KÍCH HOẠT LIỆT KÊ NGÀNH (CHỈ DÙNG KHI CÓ YÊU CẦU RÕ RÀNG):
                + NẾU VÀ CHỈ NẾU người dùng hỏi TRỰC TIẾP đến "Ngành nào", "Danh sách ngành", "Chỉ tiêu cụ thể", bạn MỚI ĐƯỢC PHÉP liệt kê ngành. Lúc này, BẮT BUỘC rà soát toàn bộ bảng, liệt kê ĐẦY ĐỦ TẤT CẢ các ngành. TUYỆT ĐỐI KHÔNG bỏ sót hoặc tự ý tóm tắt.
                + NGƯỢC LẠI, nếu người dùng CHỈ hỏi về "Hệ đào tạo" hoặc "Phương thức", CẤM liệt kê danh sách ngành bên trong hệ đó để tránh dài dòng. Chỉ liệt kê Tên Hệ/Phương thức là đủ.
            - XỬ LÝ TÌNH HUỐNG HỎI CHUNG CHUNG (AMBIGUITY): Nếu người dùng hỏi chung chung mà context có nhiều hệ đào tạo:
                + CHỈ liệt kê các hệ đào tạo MÀ CONTEXT CÓ DỮ LIỆU THỰC TẾ.
                + TUYỆT ĐỐI không tự suy ra hoặc thêm vào các hệ không có trong context.
                + KHÔNG liệt kê hệ với nội dung "không có thông tin" — nếu không có data thì bỏ qua hoàn toàn, không đề cập.
                + Mỗi mã xét tuyển chỉ được gán đúng 1 hệ duy nhất theo heading trong context.
            - Khi trả lời về một đơn vị/phòng ban cụ thể, CHỈ sử dụng thông tin
                từ chunk có heading KHỚP CHÍNH XÁC với tên đơn vị được hỏi.
                KHÔNG lấy thông tin (website, email, SĐT) từ chunk của đơn vị khác
                dù tên có vẻ tương tự.
            - Hãy phân biệt rõ thực thể 'Trường' (toàn trường/cơ sở chính/UTC/GHA) và 'Phân hiệu tại TP.HCM/UTC2/GSA'. Nếu tài liệu chỉ nói chung về 'Trường' thì KHÔNG ĐƯỢC gán đó là của Phân hiệu.
                Khi trả lời câu hỏi liên quan đến một campus cụ thể (GSA hoặc GHA), chỉ sử dụng thông tin có ghi rõ mã trường tương ứng trong context.
                Nếu một đoạn context chứa cả thông tin GSA lẫn GHA, chỉ lấy phần 
                có ghi đúng mã trường mà người dùng hỏi, bỏ qua phần còn lại.
                Không tự suy luận hay ghép thông tin từ campus khác sang.
            - QUY TẮC ĐỌC SỐ LIỆU TUYỂN SINH (CHỐNG NHIỄU LỊCH SỬ):
                + Khi người dùng hỏi về "chỉ tiêu dự kiến", "điểm chuẩn" hoặc các số liệu của "năm nay", bạn phải quét Context và tuân thủ NGHIÊM NGẶT nguyên tắc sau:
                    1. BẮT BUỘC lấy con số ở các bảng theo mốc thời gian mà người dùng muốn.
            - Ghi rõ năm của dữ liệu mà bạn lấy được từ heading trong context (VD: "Đề án tuyển sinh 2026").
    KHÔNG tự suy đoán hoặc copy năm từ câu hỏi của người dùng nếu context không xác nhận.
            - Hệ đào tạo: đọc heading context để xác định đúng hệ (Chính quy, Vừa làm vừa học,
    Liên thông, Từ xa...) rồi nêu rõ trong câu trả lời.
            - LƯU Ý QUAN TRỌNG: Phải phân biệt rõ ràng giữa số liệu của "Nhóm ngành" (tổng của nhiều ngành) và số liệu của một "Ngành" cụ thể. 
            TUYỆT ĐỐI KHÔNG lấy chỉ tiêu tổng của cả một "Nhóm ngành" để gán cho một "Ngành" đơn lẻ.
            Nếu người dùng hỏi 1 ngành cụ thể nhưng tài liệu chỉ có số liệu tổng của Nhóm ngành, bạn phải trả lời rõ: "Tài liệu hiện tại chỉ thống kê chỉ tiêu tổng của cả Nhóm ngành [Tên nhóm ngành] là [Số lượng], chưa có số liệu bóc tách chi tiết cho riêng ngành bạn hỏi."`
            - GỢI Ý CHÉO ĐỂ HIỂU ĐÚNG Ý NGƯỜI DÙNG: Học sinh thường nhầm lẫn giữa "Thủ tục nộp hồ sơ" và "Phương thức xét tuyển (cách tính điểm)". 
              + Nếu người dùng hỏi về "cách nộp hồ sơ/thủ tục đăng ký", sau khi trả lời thủ tục, BẮT BUỘC phải gợi ý thêm: "Bạn có muốn mình cung cấp thêm thông tin chi tiết về các Phương thức xét tuyển (PT1, PT2...) và cách tính điểm không?".
              + Ngược lại, nếu họ hỏi "Phương thức xét tuyển", sau khi trả lời xong hãy gợi ý: "Bạn có muốn biết thêm về thời gian và thủ tục nộp hồ sơ cho các phương thức này không?".
            - Cuối câu trả lời, nếu phù hợp, hãy gợi ý một chủ đề liên quan đến context
                mà người dùng có thể quan tâm tiếp theo (điểm chuẩn, học bổng, 
                chuyên ngành, học phí...). Thay đổi gợi ý theo ngữ cảnh câu hỏi, 
                không lặp lại cùng một câu mẫu. 
            - Nếu context không có data phù hợp để trả lời người dùng thì cuối câu kèm theo [[user/setaudience]]
            """
            full_response = ""
            async for chunk in self.answer_llm.astream(prompt):
                text = self._message_text(chunk)
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop
            print(full_response)
            memory.save_context({"input": message}, {"output": full_response})

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id,
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            self.update_faq_statistics(db, bot_msg.interaction_id, intent_id=intent_id)
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close()

    async def stream_response_from_qa(
        self,
        query: str,
        context: str,
        session_id: int = 1,
        user_id: int = 1,
        intent_id: int = 1,
        message: str = "",
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
                    session_id=session_id,
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
                    session_id=session_id,
                )

                db.add(user_msg)
                db.flush()

            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")

            prompt = f"""
            Bạn là chatbot tra cứu thông tin chuyên nghiệp của trường {self.university_name}.
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === CÂU TRẢ LỜI CHÍNH THỨC ===
            {context}
            === CÂU HỎI NGƯỜI DÙNG ===
            {query}
            Cách trả lời:
                - Dễ hiểu
                - Thân thiện
                - Trả lời bằng tiếng Việt
                - Dùng ngôn ngữ đời thường
                - Dùng Markdown linh hoạt: chỉ dùng tiêu đề ## và gạch đầu dòng khi câu trả lời có nhiều mục hoặc nhiều ý rõ ràng. Câu trả lời ngắn thì viết thành đoạn văn tự nhiên, không cần chia heading.
                - Nếu trong câu trả lời có đường dẫn thì hãy markdown đường dẫn
            Không được:
                - Lặp lại ý người dùng
                - Dùng ngôn ngữ AI máy móc, robot
            Cách phản hồi:
                - Nội dung cốt lõi BẮT BUỘC phải dựa trên [CÂU TRẢ LỜI CHÍNH THỨC]. Tuyệt đối không tự bịa số liệu, quy chế hay thông tin sai lệch.
                - Tuy nhiên, bạn được phép DIỄN ĐẠT LẠI (paraphrase) câu trả lời đó sao cho tự nhiên, mượt mà và giống con người hơn. KHÔNG cần copy/paste y chang từng chữ một cách máy móc.
                - Dù [CÂU TRẢ LỜI CHÍNH THỨC] không có thông tin mở rộng, ở cuối mỗi câu trả lời, bạn VẪN PHẢI chủ động đặt một câu hỏi mở để dẫn dắt người dùng khám phá thêm.
                - Ví dụ: Nếu người dùng hỏi về "Học phí", hãy hỏi thêm "Bạn có muốn mình tư vấn thêm về các chính sách học bổng hoặc phương thức xét tuyển không?".
                - Nhấn mạnh: Chỉ HỎI để định hướng, tuyệt đối KHÔNG tự bịa ra dữ liệu chi tiết của phần gợi ý.
                - Gợi ý thông tin liên quan hữu ích
            === HƯỚNG DẪN TRẢ LỜI ===
            - Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
            - Bạn là tư vấn tuyển sinh của trường {self.university_name}, nhớ kiểm tra kĩ rõ ràng câu hỏi, nếu câu hỏi yêu cầu thông tin của một trường khác thì nói rõ là không có dữ liệu trong hệ thống hiện tại
            - Nếu câu hỏi chỉ là chào hỏi, hỏi thời tiết, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
            """
            full_response = ""
            async for chunk in self.answer_llm.astream(prompt):
                text = self._message_text(chunk)
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop

            memory.save_context({"input": message}, {"output": full_response})
            print(
                "Saved to memory. Current messages:", len(memory.chat_memory.messages)
            )

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id,
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()

            self.update_faq_statistics(db, bot_msg.interaction_id, intent_id=intent_id)
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close()

    async def stream_response_from_recommendation(
        self, user_id: int, session_id: int, query: str, message: str
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
                    session_id=session_id,
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
                    session_id=session_id,
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
            async for chunk in self.answer_llm.astream(prompt):
                text = self._message_text(chunk)
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop

            memory.save_context({"input": query}, {"output": full_response})
            print(
                "Saved to memory. Current messages:", len(memory.chat_memory.messages)
            )

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id,
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

    async def stream_response_from_NA(
        self,
        query: str,
        context: str,
        session_id: int = 1,
        user_id: int = 1,
        intent_id: int = 0,
        message: str = "",
        current_audience_id: int = None,
        current_intent_id: int = None,
        query_embedding: Optional[List[float]] = None,
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
                    session_id=session_id,
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
                    session_id=session_id,
                )
                db.add(user_msg)
                db.flush()
            print(f"Enter Nope")
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")
            audience = (
                db.query(TargetAudience)
                .filter(TargetAudience.id == current_audience_id)
                .first()
            )

            if not audience:
                raise Exception("No valid audiences found")

            filtered_audience_names = audience.present_name
            prompt = f"""
            Bạn là chatbot tra cứu thông tin {filtered_audience_names} của trường {self.university_name}.
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === CÂU TRẢ LỜI CHÍNH THỨC ===
            {context}

            === CÂU HỎI NGƯỜI DÙNG ===
            {query}
            === PHONG CÁCH TRẢ LỜI ===
            Cách trả lời:
                - Dễ hiểu
                - Thân thiện
                - Trả lời bằng tiếng Việt
                - Dùng ngôn ngữ đời thường
                - Dùng Markdown linh hoạt: chỉ dùng tiêu đề ## và gạch đầu dòng khi câu trả lời có nhiều mục rõ ràng. Câu trả lời ngắn thì viết thành đoạn văn tự nhiên, không cần chia heading.
                - Nếu trong câu trả lời có đường dẫn thì hãy markdown đường dẫn
            Không được:
                - Lặp lại ý người dùng
                - Dùng ngôn ngữ AI máy móc, robot
                - TUYỆT ĐỐI KHÔNG ĐƯỢC LẤP LIẾM, TỰ GÁN THÔNG TIN
            Cách phản hồi:
                - Trả lời trực tiếp câu hỏi
                - Nếu cần, hướng dẫn từng bước
                - Gợi ý thông tin liên quan hữu ích
                - Nếu có đường dẫn liên quan đến nội dung người dùng muốn biết, có thể gợi ý để họ tự tìm hiểu thêm, TUYỆT ĐỐI không được lấy đường dẫn không liên quan đến nội dung trả lời
            === HƯỚNG DẪN TRẢ LỜI ===
            Bạn là tầng phản hồi của chatbot tra cứu thông tin {filtered_audience_names} của trường {self.university_name} khi mà danh mục {filtered_audience_names} hiện tại không có thông tin mà người dùng cần.
            Nhiệm vụ của bạn KHÔNG phải trả lời kiến thức,
            mà là xử lý tình huống, tự tạo câu phản hồi phù hợp với CÂU HỎI NGƯỜI DÙNG khi NGỮ CẢNH ĐƯỢC CUNG CẤP
            KHÔNG PHÙ HỢP hoặc CHƯA CÓ DATA với ý định câu hỏi người dùng.
            ## Hướng xử lý
            - Hãy nói rõ danh mục {filtered_audience_names} hiện tại không có thông tin mà người dùng cần
            - Đưa ra cách giải quyết cụ thể (liên hệ phòng ban phù hợp hoặc kênh hỗ trợ chính thức)
            - Nếu có thể, gợi ý loại đơn vị cần liên hệ dựa theo trường đại học bạn đang tư vấn (ví dụ: Phòng Tổ chức Hành chính, Phòng Đào tạo...)
            === NGUYÊN TẮC BẮT BUỘC ===
            - TUYỆT ĐỐI không suy diễn thông tin từ ngữ cảnh.
            - TUYỆT ĐỐI không trả lời theo nội dung ngữ cảnh nếu không khớp rõ ràng.
            - Không bịa thông tin.
            - Không cố gắng “trả lời cho có”.
            === VIỆC BẠN PHẢI LÀM ===
            1. Nhận diện rằng nội dung hiện có KHÔNG trả lời đúng câu hỏi.
            2. Phản hồi một cách lịch sự, rõ ràng, không máy móc, tự nhiên như 1 tư vấn tuyển sinh
            3. Hướng người dùng đi đúng hướng tiếp theo.
            4. Có thể chào hỏi nếu người dùng gửi lời chào
            5. Chỉ sử dụng "đoạn hội thoại trước" để hiểu ngữ cảnh câu hỏi, không dùng "đoạn hội thoại trước" làm nguồn thông tin trả lời.
            6. Giải thích rằng hệ thống hiện chưa có dữ liệu phù hợp 
            7. Cuối câu trả lời kèm theo [[user/setaudience]]
            """
            full_response = ""
            async for chunk in self.answer_llm.astream(prompt):
                text = self._message_text(chunk)
                full_response += text
                yield text
                await asyncio.sleep(0)  # Nhường event loop

            memory.save_context({"input": message}, {"output": full_response})
            print(
                "Saved to memory. Current messages:",
                len(memory.chat_memory.messages),
            )

            # === Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id,
            )
            db.add(bot_msg)
            db.flush()
            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            self.update_faq_statistics(db, bot_msg.interaction_id, intent_id=intent_id)
            self.update_faq_statistics_for_query(
                db, user_msg.interaction_id, intent_id=intent_id
            )
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        except Exception as e:
            print(f"response NA error: {e}")
            yield "Hệ thống đang quá tải, bạn vui lòng [Thử lại] nhé."
        finally:
            db.close()

    def get_suggestion_from_training(
        db: Session, target_audience_id: int, intent_id: Optional[int] = None
    ):

        query = db.query(TrainingQuestionAnswer).filter(
            TrainingQuestionAnswer.status == "approved"
        )
        target = (
            db.query(TargetAudience)
            .filter(TargetAudience.id == target_audience_id)
            .first()
        )

        if not target:
            return []

        # filter theo audience
        query = query.filter(TrainingQuestionAnswer.target_audiences.any(target.name))

        # filter theo intent nếu có
        if intent_id is not None and intent_id != 0:
            query = query.filter(TrainingQuestionAnswer.intent_id == intent_id)

        return query.order_by(TrainingQuestionAnswer.created_at.desc()).limit(5).all()

    def add_interaction_and_faq_for_intent_0(
        self,
        full_response: str,
        session_id: int = 1,
        user_id: int = 1,
        intent_id: int = 1,
        message: str = "",
    ):
        db = SessionLocal()
        if not user_id:
            # 🧩 1. Lưu tin nhắn người dùng
            user_msg = ChatInteraction(
                message_text=message,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=False,
                sender_id=None,
                session_id=session_id,
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
                session_id=session_id,
            )
            db.add(user_msg)
            db.flush()
        bot_msg = ChatInteraction(
            message_text=full_response,
            timestamp=datetime.now(),
            rating=None,
            is_from_bot=True,
            sender_id=None,
            session_id=session_id,
        )
        db.add(bot_msg)
        db.flush()
        db.commit()
        self.update_faq_statistics_for_query(
            db, user_msg.interaction_id, intent_id=intent_id
        )

    def update_faq_statistics_for_query(
        self, db: Session, query_id: int, intent_id: int = 1
    ):

        try:
            response = (
                db.query(ChatInteraction)
                .filter(
                    ChatInteraction.interaction_id == query_id,
                    ChatInteraction.is_from_bot == False,
                )
                .first()
            )

            if not response:
                raise ValueError("Chatbot response not found")

            faq = FaqStatistics(query_from_user_id=query_id, intent_id=intent_id)
            db.add(faq)
            db.commit()

        except Exception as e:
            db.rollback()
            print(f"Error updating FaqStatistics: {e}")

    def create_training_qa(
        self,
        db: Session,
        intent_id: int,
        question: str,
        answer: str,
        target_audiences: List[str],
        created_by: int,
        is_private: Optional[bool] = False,
    ):
        qa = TrainingQuestionAnswer(
            question=question,
            answer=answer,
            intent_id=intent_id,
            target_audiences=target_audiences,
            created_by=created_by,
            is_private=is_private,
            status="draft",
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
        audiences = (
            db.query(TargetAudience)
            .filter(TargetAudience.name.in_(audience_names))
            .all()
        )
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
                        # MULTI AUDIENCE
                        "audience_ids": audience_ids,
                        "audience_names": filtered_audience_names,
                        "question_text": qa.question,
                        "answer_text": qa.answer,
                        "type": "training_qa",
                        "is_private": qa.is_private or False,
                    },
                )
            ],
        )

        # update DB
        qa.status = "approved"
        qa.approved_by = reviewer_id
        qa.approved_at = datetime.now().date()  # Convert datetime to date
        db.commit()

        return {"postgre_question_id": qa.question_id, "qdrant_question_id": point_id}

    def delete_training_qa(self, db: Session, qa_id: int, current_user: Users):

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
                            key="question_id", match=models.MatchValue(value=qa_id)
                        )
                    ]
                )
            ),
        )

        qa.deleted_by = current_user.user_id
        db.commit()

        return {"deleted_question_id": qa_id}

    def create_document(
        self,
        db: Session,
        title: str,
        file_path: str,
        intend_id: int,
        target_audiences: List[str],
        created_by: int,
        is_private: Optional[bool] = False,
        content: Optional[str] = None,
        is_ocr: bool = False,
        path_txt: Optional[str] = None,
    ):
        new_doc = KnowledgeBaseDocument(
            title=title,
            file_path=file_path,
            intend_id=intend_id,
            target_audiences=target_audiences,
            status="draft",
            is_private=is_private,
            created_by=created_by,
            content=content,
            is_ocr=is_ocr,
            path_txt=path_txt,
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        return new_doc

    def approve_document(
        self,
        db: Session,
        document_id: int,
        reviewer_id: int,
        intent_id: int,
        metadata: dict = None,
    ):

        doc = db.query(KnowledgeBaseDocument).filter_by(document_id=document_id).first()
        if not doc:
            raise Exception("Document not found")

        if doc.status != "draft":
            raise Exception("Only draft documents can be approved")
        audience_names_input = doc.target_audiences or []

        audiences = (
            db.query(TargetAudience)
            .filter(TargetAudience.name.in_(audience_names_input))
            .all()
        )

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
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".txt": "text/plain",
        }
        ext = os.path.splitext(doc.file_path)[1].lower()
        # Dùng content đã lưu trong DB (hoặc từ file txt nếu content quá dài)
        content = getattr(doc, "content", None)
        if not content:
            txt_path = getattr(doc, "path_txt", None)
            if txt_path:
                resolved = (
                    os.path.join(os.getcwd(), txt_path)
                    if not os.path.isabs(txt_path)
                    else txt_path
                )
                if os.path.exists(resolved):
                    with open(resolved, "r", encoding="utf-8") as f:
                        content = f.read()

        # Fallback: extract trực tiếp từ file nếu không có content
        if not content:
            mime_type = mime_map.get(ext, "text/plain")
            content = DocumentProcessor.extract_text(
                file_content=file_bytes,
                filename=os.path.basename(doc.file_path),
                mime_type=mime_type,
            )
        # --- Split text ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000, chunk_overlap=1000
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
                            "type": "document",
                            "is_private": doc.is_private or False,
                        },
                    )
                ],
            )

            qdrant_ids.append(point_id)

        # update document status
        doc.status = "approved"
        doc.reviewed_by = reviewer_id
        doc.reviewed_at = datetime.now().date()  # Convert datetime to date
        db.commit()

        return {"document_id": document_id, "status": doc.status}

    async def approve_document_stream(self, document_id: int, reviewer_id: int):
        """
        Async SSE generator: yield progress events while approving & indexing a document.

        Events emitted:
          {"event": "start",    "total_chunks": N}
          {"event": "progress", "chunk": N, "total": M, "progress": P}
          {"event": "done",     "document_id": ID, "status": "approved", "total_chunks": M, "qdrant_points": N}
          {"event": "error",    "message": "..."}
        """
        db = SessionLocal()
        try:
            doc = (
                db.query(KnowledgeBaseDocument)
                .filter_by(document_id=document_id)
                .first()
            )
            if not doc:
                yield f"data: {json.dumps({'event': 'error', 'message': 'Document not found'})}\n\n"
                return

            if doc.status != "draft":
                yield f"data: {json.dumps({'event': 'error', 'message': 'Only draft documents can be approved'})}\n\n"
                return

            audience_names_input = doc.target_audiences or []
            audiences = (
                db.query(TargetAudience)
                .filter(TargetAudience.name.in_(audience_names_input))
                .all()
            )
            if not audiences:
                yield f"data: {json.dumps({'event': 'error', 'message': 'No valid audiences found'})}\n\n"
                return

            audience_ids = [a.id for a in audiences]
            filtered_audience_names = [a.present_name for a in audiences]
            missing = set(audience_names_input) - {a.name for a in audiences}
            if missing:
                yield f"data: {json.dumps({'event': 'error', 'message': f'Audience not found: {missing}'})}\n\n"
                return

            intent = db.query(Intent).filter_by(intent_id=doc.intend_id).first()

            # --- Get content ---
            content = getattr(doc, "content", None)
            if not content:
                txt_path = getattr(doc, "path_txt", None)
                if txt_path:
                    resolved = (
                        os.path.join(os.getcwd(), txt_path)
                        if not os.path.isabs(txt_path)
                        else txt_path
                    )
                    if os.path.exists(resolved):
                        with open(resolved, "r", encoding="utf-8") as f:
                            content = f.read()

            if not content:
                abs_path = os.path.abspath(doc.file_path)
                mime_map = {
                    ".pdf": "application/pdf",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".txt": "text/plain",
                }
                ext = os.path.splitext(doc.file_path)[1].lower()
                with open(abs_path, "rb") as fb:
                    content = DocumentProcessor.extract_text(
                        fb.read(),
                        os.path.basename(doc.file_path),
                        mime_map.get(ext, "text/plain"),
                    )

            if not content:
                yield f"data: {json.dumps({'event': 'error', 'message': 'No content to index'})}\n\n"
                return

            # --- Split + embed chunks ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=12000, chunk_overlap=1000
            )
            chunks = text_splitter.split_text(content)
            total = len(chunks)

            yield f"data: {json.dumps({'event': 'start', 'total_chunks': total})}\n\n"

            for i, chunk in enumerate(chunks):
                progress = round((i + 1) / total * 100)
                yield f"data: {json.dumps({'event': 'progress', 'chunk': i + 1, 'total': total, 'progress': progress})}\n\n"

                loop = asyncio.get_event_loop()
                embed_future = loop.run_in_executor(
                    None, self.embeddings.embed_query, chunk
                )
                deadline = time.monotonic() + 120
                embedding = None
                while not embed_future.done():
                    if time.monotonic() > deadline:
                        embed_future.cancel()
                        break
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(8)

                if embed_future.done() and not embed_future.cancelled():
                    embedding = embed_future.result()
                else:
                    embedding = self.embeddings.embed_query(chunk)

                point_id = str(uuid.uuid4())
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
                                "audience_ids": audience_ids,
                                "audience_names": filtered_audience_names,
                                "intent_id": doc.intend_id,
                                "intent_name": intent.intent_name if intent else None,
                                "type": "document",
                                "is_private": doc.is_private or False,
                            },
                        )
                    ],
                )

            # --- Finalize ---
            doc.status = "approved"
            doc.reviewed_by = reviewer_id
            doc.reviewed_at = datetime.now()
            db.commit()

            yield f"data: {json.dumps({'event': 'done', 'document_id': document_id, 'status': 'approved', 'total_chunks': total})}\n\n"

        except Exception as e:
            db.rollback()
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        finally:
            db.close()
    def _get_char_splitter(self, content: str):
        """
        Trả về char_splitter với chunk_size phù hợp theo nội dung thực tế.
        - Nhiều bảng (tuyển sinh): giữ chunk lớn để không mất context bảng
        - Document ngắn (KTX, thông báo...): chunk nhỏ để signal không bị loãng
        - Document trung bình: chunk vừa
        """
        table_count = content.count("| --- |")
        total_chars = len(content)

        if table_count >= 5:
            # Tài liệu tuyển sinh nhiều bảng → giữ nguyên logic cũ
            chunk_size, chunk_overlap = 12000, 1000
        elif total_chars < 3000:
            # Document ngắn: chia ~5 chunks, tối thiểu 300 ký tự/chunk
            chunk_size = max(300, total_chars // 5)
            chunk_overlap = 50
        elif total_chars < 10000:
            chunk_size, chunk_overlap = 1000, 100
        else:
            chunk_size, chunk_overlap = 3000, 200

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ";", ",", " "],
        )
    def _extract_and_chunk(self, doc, ext: str) -> tuple[list[str], bool]:
        """
        Extract content từ file và chunk có header-context.
        Chạy trong executor (blocking IO).

        Returns:
            (chunks, use_header_split)
            use_header_split = True nếu đã dùng MarkdownHeaderTextSplitter
        """

        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000, chunk_overlap=1000
        )
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )

        # --- Bước 1: Lấy raw bytes ---
        file_bytes = self._read_file_bytes(doc, ext)
        print(f"[DEBUG] ext={ext}")
        print(f"[DEBUG] file_bytes length={len(file_bytes)}")
        print(f"[DEBUG] file_bytes[:20]={file_bytes[:20]}")
        # --- Bước 2: Route theo loại file ---

        # DOCX — extract_text_from_docx đã trả về markdown
        if ext in (".docx", ".doc"):
            markdown_content = DocumentProcessor.extract_text_from_docx_2(file_bytes)
            print(f"[DEBUG] markdown_content[:500]=\n{markdown_content[:500]}")
            has_headings = any(
                line.startswith("#") for line in markdown_content.splitlines()
            )
            char_splitter = self._get_char_splitter(markdown_content)
            print(f"[DEBUG] has_headings={has_headings}")

            header_docs = header_splitter.split_text(markdown_content)
            print(f"[DEBUG] header_docs count={len(header_docs)}")
            print(
                f"[DEBUG] header_docs[0].metadata={header_docs[0].metadata if header_docs else 'empty'}"
            )
            ...
            return (
                self._header_chunks(markdown_content, header_splitter, char_splitter),
                True,
            )

        # PDF — thử pdfplumber (có heading), fallback OCR (plain text)
        if ext == ".pdf":
            markdown_content = DocumentProcessor.extract_text_from_pdf_2(
                file_bytes, os.path.basename(doc.file_path)
            )
            # Docling luôn trả về markdown kể cả PDF scan
            # chỉ cần check có heading không để quyết định splitter
            has_headings = any(
                line.startswith("#") for line in markdown_content.splitlines()
            )
            char_splitter = self._get_char_splitter(markdown_content)
            print(f"[DEBUG] chunk_size={char_splitter._chunk_size}")
            if has_headings:
                return (
                    self._header_chunks(
                        markdown_content, header_splitter, char_splitter
                    ),
                    True,
                )
            return self._plain_chunks(markdown_content, char_splitter), False

        # TXT — plain text, không có heading
        if ext == ".txt":
            content = file_bytes.decode("utf-8", errors="ignore")
            content = DocumentProcessor.clean_text(content)
            char_splitter = self._get_char_splitter(content)
            return self._plain_chunks(content, char_splitter), False

        # Các định dạng còn lại (xlsx, xls, pptx, html)
        mime_map = {
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".html": "text/html",
        }
        content = DocumentProcessor.extract_text(
            file_bytes,
            os.path.basename(doc.file_path),
            mime_map.get(ext, "text/plain"),
        )
        char_splitter = self._get_char_splitter(content)
        return self._plain_chunks(content, char_splitter), False

    def _read_file_bytes(self, doc, ext: str) -> bytes:

        if ext in (".docx", ".doc", ".pdf"):
            abs_path = os.path.abspath(doc.file_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"File gốc không tìm thấy: {abs_path}")
            with open(abs_path, "rb") as f:
                return f.read()

        # Các định dạng khác: dùng content đã lưu nếu có
        if getattr(doc, "content", None):
            if isinstance(doc.content, bytes):
                return doc.content
            return doc.content.encode("utf-8")

        txt_path = getattr(doc, "path_txt", None)
        if txt_path:
            resolved = (
                txt_path
                if os.path.isabs(txt_path)
                else os.path.join(os.getcwd(), txt_path)
            )
            if os.path.exists(resolved):
                with open(resolved, "rb") as f:
                    return f.read()

        abs_path = os.path.abspath(doc.file_path)
        with open(abs_path, "rb") as f:
            return f.read()

    def _header_chunks(
        self, markdown: str, header_splitter, char_splitter
    ) -> list[str]:
        """Chunk có header context — dùng cho DOCX và PDF có heading."""
        header_docs = header_splitter.split_text(markdown)
        split_docs = char_splitter.split_documents(header_docs)

        chunks = []
        for doc_chunk in split_docs:
            headers = [
                doc_chunk.metadata[k]
                for k in ["h1", "h2", "h3"]
                if k in doc_chunk.metadata
            ]
            text = doc_chunk.page_content.strip()
            if not text:
                continue
            if headers:
                text = f"[{' > '.join(headers)}]\n{text}"
            chunks.append(text)
        result = []
        for chunk in chunks:
            result.extend(self._split_large_table(chunk, max_rows=3))
        return result

    def _plain_chunks(self, content: str, char_splitter) -> list[str]:
        """Chunk plain text — dùng cho TXT, OCR, xlsx, pptx..."""
        return [c for c in char_splitter.split_text(content) if c.strip()]
    def _merge_wrapped_lines(self, lines: list[str]) -> list[str]:
        """
        Merge dòng bị word-wrap: nếu dòng hiện tại match TITLE_PATTERN
        nhưng dòng tiếp theo là 1 từ đơn không match pattern nào
        → đây là phần còn lại của tên bị wrap, merge vào.
        """
        SINGLE_WORD = re.compile(r'^\S+$')
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Nếu dòng này là tên người (match title pattern)
            if self.TITLE_PATTERN.match(line):
                # Peek dòng tiếp: nếu là 1 từ đơn và không phải title mới → merge
                while (
                    i + 1 < len(lines)
                    and SINGLE_WORD.match(lines[i + 1])
                    and not self.TITLE_PATTERN.match(lines[i + 1])
                ):
                    i += 1
                    line = line + ' ' + lines[i]
            result.append(line)
            i += 1
        return result
    def _restructure_personnel_blocks(
    self, chunks: list[str], unit_name: str
) -> list[str]:
        """
        Post-process chunks sau khi split:
        - Detect block nhân sự bị merge cột
        - Restructure thành text rõ ràng từng người
        - Các chunk không phải nhân sự: giữ nguyên
        """
        return [
            self._try_restructure_chunk(chunk, unit_name) 
            for chunk in chunks
        ]

    def _try_restructure_chunk(self, chunk: str, unit_name: str) -> str:
        """
        Thử restructure một chunk.
        Nếu không nhận diện được pattern → trả về nguyên bản.
        """
        if self._is_personnel_block(chunk):
            return self._restructure_personnel(chunk, unit_name)
        # Thêm các loại khác tại đây trong tương lai:
        # if self._is_schedule_block(chunk):
        #     return self._restructure_schedule(chunk)
        return chunk

    # ============================================================
    # PERSONNEL BLOCK
    # ============================================================

    TITLE_PATTERN = re.compile(
    r'^(GS\.?\s*|PGS\.?\s*)?(TS|ThS|Ths|ThS\.NCS|Ths\.NCS|KS|CN|NCS)\.?\s+\S',
    re.MULTILINE)

    def _restructure_personnel(self, raw: str, unit_name: str) -> str:
        print(f"[DEBUG RESTRUCTURE INPUT]\n{repr(raw[:500])}")
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        print(f"[DEBUG LINES] {lines[:10]}")
        lines = self._merge_wrapped_lines(lines)
        tokens = []
        for line in lines:
            if self.TITLE_PATTERN.match(line):
                tokens.append(('name', line))
            else:
                # KHÔNG split ' | ' ở đây — giữ nguyên cả dòng làm 1 role
                tokens.append(('role', line))

        persons = []
        i = 0
        while i < len(tokens):
            if tokens[i][0] == 'name':
                names = []
                while i < len(tokens) and tokens[i][0] == 'name':
                    names.append({'name': tokens[i][1], 'roles': []})
                    i += 1

                n = len(names)
                role_idx = 0
                while i < len(tokens) and tokens[i][0] == 'role':
                    role_val = tokens[i][1]

                    # Heuristic: role quá ngắn (< 4 ký tự) và không có động từ/danh từ chức vụ
                    # → gắn vào tên của person trước
                    if (
                        len(role_val) < 5
                        and not any(kw in role_val for kw in [
                            'viên', 'trưởng', 'phó', 'giám', 'ban', 'phòng'
                        ])
                        and role_idx < n  # chỉ check khi chưa qua vòng đầu
                    ):
                        names[role_idx % n]['name'] += ' ' + role_val
                    else:
                        names[role_idx % n]['roles'].append(role_val)
                        role_idx += 1

                    i += 1

                persons.extend(names)
            else:
                i += 1

        if not persons:
            return raw

        out = []
        for p in persons:
            roles = [r for r in p['roles'] if r]
            parts = [p['name']] + roles
            out.append(' | '.join(parts))

        return '\n'.join(out)

    def _is_personnel_block(self, text: str) -> bool:
        """
        Heuristic: có ít nhất 3 tên người → khả năng cao là bảng nhân sự.
        Tránh false positive với văn bản thường nhắc đến 1-2 người.
        """
        matches = self.TITLE_PATTERN.findall(text)
        print(f"[DEBUG IS_PERSONNEL] matches={matches} count={len(matches)}")
        return len(matches) >= 3
    def _flatten_table_row(self, row: str) -> str:
        """
        Chuyển 1 markdown table row thành plain text.
        | 4 | Phòng CTCT&SV | Trưởng phòng: ThS. Đặng Văn Ơn... |
        → Phòng CTCT&SV: Trưởng phòng: ThS. Đặng Văn Ơn...
        """
        # Tách cells, bỏ cell đầu (số thứ tự) nếu là số
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if not cells:
            return ""
        
        # Bỏ cell đầu nếu là số thứ tự
        if cells[0].isdigit():
            cells = cells[1:]
        
        if not cells:
            return ""
        
        # Cell đầu là tên đơn vị/người, các cell sau là nội dung
        if len(cells) == 1:
            return cells[0]
        
        subject = cells[0]
        content = " ".join(cells[1:])
        return f"{subject}: {content}"
    
    def _split_large_table(self, chunk: str, max_rows: int = 10) -> list[str]:
        lines = chunk.splitlines()
        header_context = []
        table_lines = []
        in_table = False

        for line in lines:
            if in_table and line.strip() == "":
                in_table = False
            if line.startswith("|") and not in_table:
                in_table = True
            if in_table:
                table_lines.append(line)
            else:
                header_context.append(line)

        if not table_lines:
            return [chunk]

        table_header = []
        data_rows = []
        found_separator = False
        for line in table_lines:
            if not found_separator:
                table_header.append(line)
                if re.search(r'\|\s*---', line):  # fix vấn đề 2
                    found_separator = True
            else:
                if line.startswith("| TT ") or line.startswith("| tt "):
                    table_header.append(line)
                else:
                    data_rows.append(line)
        print(f"[DEBUG SPLIT_TABLE] data_rows={len(data_rows)} max_rows={max_rows} will_split={len(data_rows) > max_rows}")
        if len(data_rows) <= max_rows:
            # Flatten luôn dù không split
            prefix = "\n".join(header_context)
            flat_rows = [self._flatten_table_row(r) for r in data_rows if r.strip()]
            flat_rows = [r for r in flat_rows if r]
            return [f"{prefix}\n" + "\n".join(flat_rows)] if flat_rows else [chunk]

        prefix = "\n".join(header_context)
        table_head_str = "\n".join(table_header)

        sub_chunks = []
        for i in range(0, len(data_rows), max_rows):
            batch = data_rows[i: i + max_rows]
            label = f"{prefix} (tiếp theo)\n" if sub_chunks else f"{prefix}\n"
            flat_rows = [self._flatten_table_row(r) for r in batch if r.strip()]
            flat_rows = [r for r in flat_rows if r]
            
            if flat_rows:
                sub_chunks.append(label + "\n".join(flat_rows))

        return sub_chunks if sub_chunks else [chunk]

    def _enrich_table_chunks(self, chunks: list[str], doc_context: str) -> list[str]:
        def has_table(chunk: str) -> bool:
            return "| --- |" in chunk or "|---|" in chunk

        enriched = []
        for chunk in chunks:
            if not has_table(chunk):
                enriched.append(chunk)
                continue

            # --- THÊM: split bảng lớn trước ---
            sub_chunks = self._split_large_table(chunk, max_rows=10)

            for sub in sub_chunks:
                try:
                    prompt = f"""Bạn là trợ lý phân tích tài liệu của trường đại học.

                    Dưới đây là một đoạn trích từ tài liệu có chứa bảng dữ liệu:
                    ---
                    {sub}
                    ---

                    Ngữ cảnh tài liệu:
                    {doc_context[:1500]}

                    Hãy viết một đoạn mô tả ngắn (3-5 câu) bằng tiếng Việt giúp hệ thống tìm kiếm 
                    nhận diện đúng nội dung bảng này. Mô tả cần:

                    1. Bắt đầu bằng: "Bảng này thuộc [tên mục/section đầy đủ lấy từ dòng [...] 
                    trong đoạn trích], áp dụng cho [đối tượng], tại [địa điểm/đơn vị nếu có]."
                    Lưu ý: dùng đúng tên section trong dòng [...], không tự đặt lại.

                    2. Tóm tắt NỘI DUNG CHÍNH của bảng — liệt kê các cột/trường thông tin quan trọng 
                    và các giá trị nổi bật (ví dụ: thời hạn, mức tiền, tên ngành, chức danh, 
                    số lượng, điểm số... tùy loại bảng).

                    3. Nêu 1-2 thông tin cụ thể quan trọng nhất trong bảng 
                    (con số, mốc thời gian, tên đơn vị...) để giúp tìm kiếm chính xác hơn.

                    TUYỆT ĐỐI KHÔNG nhận xét về thông tin bị thiếu hay không có trong bảng.
                    Chỉ mô tả những gì THỰC SỰ CÓ trong đoạn trích.
                    Chỉ trả về đoạn mô tả, không giải thích thêm."""

                    response = self.llm.invoke(prompt)
                    description = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                    enriched.append(f"{description.strip()}\n\n{sub}")
                except Exception as e:
                    print(f"[WARN] Enrich table chunk failed: {e}")
                    enriched.append(sub)

        return enriched

    def delete_document(self, db: Session, document_id: int, current_user: Users):
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
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

        # Xóa chunks trong DB
        dl = db.query(DocumentChunk).filter_by(document_id=document_id)
        if dl:
            dl.delete()

        doc.deleted_by = current_user.user_id
        db.commit()

        return {"deleted_document_id": document_id}

    def add_document(
        self, document_id: int, content: str, intend_id: int, metadata: dict = None
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000,  # Size optimal cho Vietnamese
            chunk_overlap=1000,  # Overlap to preserve context
        )
        chunks = text_splitter.split_text(content)

        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Embed chunk
            embedding = self.embeddings.embed_query(chunk)
            point_id = str(uuid.uuid4())

            # Upsert to Qdrant
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
                            "intend_id": intend_id,
                            "metadata": metadata or {},
                            "type": "document",
                        },
                    )
                ],
            )
            chunk_ids.append(point_id)

        return chunk_ids

    def get_deleted_questions(self, db: Session):
        results = (
            db.query(TrainingQuestionAnswer)
            .filter(TrainingQuestionAnswer.status == "deleted")
            .all()
        )

        response = []

        for item in results:
            response.append(
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "answer": item.answer,
                    "intent_id": item.intent.intent_id if item.intent else None,
                    "intent_name": item.intent.intent_name if item.intent else None,
                    "status": item.status,
                    "created_at": item.created_at,
                    "approved_at": item.approved_at,
                    "created_by": item.created_by,
                    "approved_by": item.approved_by,
                    "created_by_name": (
                        item.created_by_user.full_name if item.created_by_user else None
                    ),
                    "approved_by_name": (
                        item.approved_by_user.full_name
                        if item.approved_by_user
                        else None
                    ),
                    "deleted_by": item.deleted_by,
                    "deleted_by_name": (
                        item.deleted_by_user.full_name if item.deleted_by_user else None
                    ),
                    "reject_reason": getattr(item, "reject_reason", None),
                    "target_audiences": getattr(item, "target_audiences", []),
                }
            )

        return response

    def get_deleted_documents(self, db: Session):
        results = (
            db.query(KnowledgeBaseDocument)
            .filter(KnowledgeBaseDocument.status == "deleted")
            .all()
        )

        response = []

        for item in results:
            response.append(
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "file_path": item.file_path,
                    "category": item.category,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "created_by": item.created_by,
                    "created_by_name": (item.author.full_name if item.author else None),
                    "reviewed_by": item.reviewed_by,
                    "reviewed_by_name": (
                        item.reviewer.full_name if item.reviewer else None
                    ),
                    "reviewed_at": item.reviewed_at,
                    "deleted_by": item.deleted_by,
                    "deleted_by_name": (
                        item.deleter.full_name if item.deleter else None
                    ),
                    "reject_reason": getattr(item, "reject_reason", None),
                    "target_audiences": getattr(item, "target_audiences", []),
                    "content": item.content,
                    "is_ocr": item.is_ocr,
                    "path_txt": item.path_txt,
                    "status": item.status,
                    "intent_id": item.intent.intent_id if item.intent else None,
                    "intent_name": item.intent.intent_name if item.intent else None,
                }
            )

        return response

    async def cross_scope_search(
        self, query: str, top_k: int = 3, query_embedding: Optional[List[float]] = None
    ):
        if query_embedding is None:
            query_embedding = await self.embeddings.aembed_query(query)

        # ===== 1. SEARCH TRAINING QA =====
        qa_results = await self.async_qdrant_client.search(
            collection_name=self.training_qa_collection,
            query_vector=query_embedding,
            limit=top_k,
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
                "score": top.score,
            }

        # ===== 2. SEARCH DOCUMENT =====
        doc_results = await self.async_qdrant_client.search(
            collection_name=self.documents_collection,
            query_vector=query_embedding,
            limit=top_k,
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
                "score": top.score,
            }

        return None

    async def cross_scope_search_score(
        self,
        query: str,
        top_k: int = 3,
        query_embedding: Optional[List[float]] = None,
        current_audience_id: int = None,
        current_intent_id: int = None,
    ):
        # Biến đổi câu hỏi thành vector
        if query_embedding is None:
            query_embedding = await self.embeddings.aembed_query(query)
        candidates = []

        # ===== 1. RETRIEVE TỪ CẢ 2 NGUỒN (GOM THÔ) =====
        # Lấy từ training_qa
        qa_results = await self.async_qdrant_client.search(
            collection_name=self.training_qa_collection,
            query_vector=query_embedding,
            limit=top_k,
            # Hạ threshold xuống thấp để không bỏ sót các câu có tiềm năng
            score_threshold=0.3,
        )
        for hit in qa_results:
            q_text = hit.payload.get("question_text", "")
            a_text = hit.payload.get("answer_text", "")
            combined_text = f"question: {q_text}, answer: {a_text}"
            candidates.append(
                {
                    "source": "training_qa",
                    "text": combined_text,  # Hoặc field chứa cả câu hỏi+trả lời
                    "payload": hit.payload,
                    "vector_score": hit.score,
                }
            )

        # Lấy từ document
        doc_results = await self.async_qdrant_client.search(
            collection_name=self.documents_collection,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.3,
        )
        for hit in doc_results:
            candidates.append(
                {
                    "source": "document",
                    "text": hit.payload.get("chunk_text", ""),
                    "payload": hit.payload,
                    "vector_score": hit.score,
                }
            )
        print("=== CANDIDATES BEFORE FILTER ===")
        for c in candidates:
            print(
                {
                    "source": c["source"],
                    "audience_ids": c["payload"].get("audience_ids"),
                    "intent_id": c["payload"].get("intent_id"),
                    "vector_score": c["vector_score"],
                    "text_preview": c["text"][:80],
                }
            )
        print(
            f"current_audience_id={current_audience_id}, current_intent_id={current_intent_id}"
        )
        print("=== CANDIDATES AFTER FILTER ===")
        # Nếu không có ứng viên nào vượt qua ngưỡng 0.3 của vector, dừng luôn
        if not candidates:
            return None

        def is_same_scope(payload):
            doc_audience_ids = payload.get("audience_ids") or []
            if not isinstance(doc_audience_ids, list):
                doc_audience_ids = [doc_audience_ids]

            if current_audience_id in doc_audience_ids:
                return True  # Cùng audience nhưng khác intent → GIỮ LẠI

            return False  # Hoàn toàn cùng scope → LOẠI

        candidates = [c for c in candidates if not is_same_scope(c["payload"])]
        if not candidates:
            return None
        # ===== 2. CROSS-ENCODER RE-RANKING =====
        # Tạo danh sách các cặp: [[Câu hỏi, Text ứng viên 1], [Câu hỏi, Text ứng viên 2], ...]
        sentence_pairs = [[query, cand["text"]] for cand in candidates]

        # Model chấm điểm đồng loạt
        rerank_scores = RERANKER_MODEL.predict(sentence_pairs, batch_size=16)

        # Cập nhật điểm mới vào mảng candidates
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(rerank_scores[i])

        # Sắp xếp danh sách dựa trên điểm Re-rank (từ cao xuống thấp)
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # ===== 3. CHỌN NGƯỜI CHIẾN THẮNG =====
        best_match = candidates[0]

        # LƯU Ý VỀ ĐIỂM SỐ CỦA BGE-RERANKER:
        # Điểm trả về không phải từ 0-1 mà là Logit (thường chạy từ -10 đến 10).
        # Điểm > 0 thường là có liên quan, điểm âm (< 0) là không liên quan.
        print(f" Điểm Rerank cao nhất đang là: {best_match['rerank_score']}")
        print(f" Text đang xét: {best_match['text']}")
        cross_score = float(os.getenv("CROSS_ENCODER_SCORE", 0.6))
        for c in candidates:
            print(
                {
                    "source": c["source"],
                    "audience_ids": c["payload"].get("audience_ids"),
                    "intent_id": c["payload"].get("intent_id"),
                    "rerank_score": c.get("rerank_score"),
                }
            )
        if best_match["rerank_score"] >= cross_score:
            payload = best_match["payload"]

            if best_match["source"] == "training_qa":
                return {
                    "source": "training_qa",
                    "audience_ids": payload.get("audience_ids"),
                    "audience_names": payload.get("audience_names"),
                    "intent_id": payload.get("intent_id"),
                    "intent_name": payload.get("intent_name"),
                    "question": payload.get("question_text"),
                    "score": best_match["vector_score"],
                }
            else:
                return {
                    "source": "document",
                    "audience_ids": payload.get("audience_ids"),
                    "audience_names": payload.get("audience_names"),
                    "intent_id": payload.get("intent_id"),
                    "intent_name": payload.get("intent_name"),
                    "question": payload.get("question_text"),
                    "chunk_preview": payload.get("chunk_text", "")[:200],
                    "score": best_match["vector_score"],
                }

        # Nếu Top 1 sau khi rerank vẫn có điểm < 0, coi như không tìm thấy đáp án hợp lệ
        return None

    def add_training_qa(
        self, db: Session, intent_id: int, question_text: str, answer_text: str
    ):
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
            status="draft",  # New Q&A starts as draft, needs review before training
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
                        "type": "training_qa",
                    },
                )
            ],
        )

        return {
            "postgre_question_id": new_qa.question_id,
            "qdrant_question_id": point_id,
        }

    async def search_documents(
        self,
        query: str,
        audience_ids: int,
        intent_id: int = None,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        stage: str = "document_search",
        query_embedding: Optional[List[float]] = None,
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

        must_conditions = [{"key": "audience_ids", "match": {"value": audience_ids}}]
        if intent_id:
            must_conditions.append({"key": "intent_id", "match": {"value": intent_id}})

        start = time.perf_counter()
        self._debug_log(
            f"{stage}: start search_documents top_k={top_k} query_len={len(query or '')}",
            trace_id,
        )
        
        try:
            if query_embedding is None:
                query_embedding = await self.embeddings.aembed_query(query)

            raw_results = await self.async_qdrant_client.search(
                collection_name=self.documents_collection,
                query_vector=query_embedding,
                limit=top_k,
                
                query_filter={"must": must_conditions},
            )

            if not raw_results:
                return []
            
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            top_score = (
                float(getattr(raw_results[0], "score", 0.0)) if raw_results else 0.0
            )
            top_payload = getattr(raw_results[0], "payload", {}) if raw_results else {}
            top_document_id = (top_payload or {}).get("document_id")
            self._debug_log(
                f"{stage}: success results={len(raw_results)} top_score={top_score:.6f} "
                f"top_document_id={top_document_id} elapsed_ms={elapsed_ms}",
                trace_id,
            )
            return raw_results
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._debug_log(
                f"{stage}: search_documents error type={type(e).__name__} "
                f"message={e} elapsed_ms={elapsed_ms}",
                trace_id,
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

    def has_private_content(self, chunks: list) -> bool:
        return any(
            chunk.payload.get("is_private", False)
            for chunk in chunks
            if hasattr(chunk, "payload")
        )

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
                audience_ids = (top_match.payload.get("audience_ids"),)
                audience_names = (top_match.payload.get("audience_names"),)
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
            snippet_lines.append(f"{idx}. [DOC_ID={doc_id}] {chunk_text}")

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
            res = await self.control_llm.ainvoke(prompt)
            raw = self._message_text(res).strip()
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
                trace_id,
            )
            return final_ids
        except Exception as e:
            self._debug_log(
                f"citation_guard: error type={type(e).__name__} message={e}", trace_id
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

    async def search_training_qa(
        self,
        query: str,
        audience_ids: int,
        intent_id: int = None,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        stage: str = "training_qa_search",
        query_embedding: Optional[List[float]] = None,
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

        must_conditions = [{"key": "audience_ids", "match": {"value": audience_ids}}]
        if intent_id:
            must_conditions.append({"key": "intent_id", "match": {"value": intent_id}})

        start = time.perf_counter()
        self._debug_log(
            f"{stage}: start search_training_qa top_k={top_k} query_len={len(query or '')}",
            trace_id,
        )

        try:
            if query_embedding is None:
                query_embedding = await self.embeddings.aembed_query(query)

            results = await self.async_qdrant_client.search(
                collection_name=self.training_qa_collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter={"must": must_conditions},
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            top_score = float(getattr(results[0], "score", 0.0)) if results else 0.0
            top_payload = getattr(results[0], "payload", {}) if results else {}
            top_question_id = (top_payload or {}).get("question_id")
            self._debug_log(
                f"{stage}: success results={len(results)} top_score={top_score:.6f} "
                f"top_question_id={top_question_id} elapsed_ms={elapsed_ms}",
                trace_id,
            )
            return results
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._debug_log(
                f"{stage}: search_training_qa error type={type(e).__name__} "
                f"message={e} elapsed_ms={elapsed_ms}",
                trace_id,
            )
            print(f"Qdrant search_training_qa timeout/error: {e}")
            return []

    async def hybrid_search(
        self,
        audience_ids: int,
        query: str,
        intent_id: int = None,
        trace_id: Optional[str] = None,
    ):
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
        confidence_score = float(os.getenv("CONFIDENCE_SCORE", 0.35))
        # STEP 1: Search training Q&A
        self._debug_log(f"hybrid_search: start query_len={len(query or '')}", trace_id)
        top_k = self.top_k
        # if audience_ids == 4:
        #     top_k = 10
        # Optimize: Embed query once
        try:
            query_embedding = await self.embeddings.aembed_query(query)
        except Exception as e:
            self._debug_log(f"hybrid_search: embedding error {e}", trace_id)
            query_embedding = None

        qa_results = await self.search_training_qa(
            query,
            audience_ids,
            intent_id,
            top_k=top_k,
            trace_id=trace_id,
            stage="hybrid_training_qa_search",
            query_embedding=query_embedding,
        )
        if qa_results:
            print("qa result " + qa_results[0].payload.get("answer_text"))
            print(f"score: + {qa_results[0].score}")
            self._debug_log(
                f"hybrid_search: qa_results={len(qa_results)} top_score={qa_results[0].score:.6f}",
                trace_id,
            )
        else:
            self._debug_log("hybrid_search: qa_results=0", trace_id)

        # TIER 1: Perfect match (score > 0.7)
        if qa_results and qa_results[0].score >= confidence_score:
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
                "sources": [],
                "query_embedding": query_embedding,
            }

        # TIER 2: No training Q&A match, try documents

        doc_results = await self.search_documents(
            query,
            audience_ids,
            intent_id,
            top_k=top_k,
            trace_id=trace_id,
            stage="hybrid_tier2_document_search",
            query_embedding=query_embedding,
        )
        result = self.build_document_search_result(doc_results)
        result["query_embedding"] = query_embedding
        self._debug_log(
            f"hybrid_search: tier=document confidence={result.get('confidence', 0.0):.6f} "
            f"sources={len(result.get('sources', []))}",
            trace_id,
        )
        return result

    def _get_user_personality_and_academics(
        self, user_id: int, db: Session
    ) -> Dict[str, Any]:
        out = {
            "personality_summary": None,
            "riasec": None,
            "academic_summary": None,
            "gpa": None,
            "subjects": {},
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
            out["personality_summary"] = ri.result or self._riasec_to_summary(
                out["riasec"]
            )

        # --- Academic scores ---
        score = (
            db.query(AcademicScore).filter(AcademicScore.customer_id == user_id).first()
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
            out["academic_summary"] = f"GPA xấp xỉ {gpa}. Các môn: " + ", ".join(
                [f"{k}: {v}" for k, v in subj_map.items()]
            )
            print(out["academic_summary"])
        return out

    def _riasec_to_summary(self, ri_map: Dict[str, int]) -> str:
        # very small helper - bạn có thể mở rộng
        order = sorted(ri_map.items(), key=lambda x: -x[1])
        top = order[0][0] if order else None
        return f"Ưu thế RIASEC: {', '.join([f'{k}={v}' for k,v in ri_map.items()])}. Chính: {top}."

    def _get_all_majors_from_db(
        self, db: Session, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Lấy danh sách majors
        """
        rows = db.query(Major).order_by(Major.major_name).limit(limit).all()
        majors = []
        for r in rows:
            majors.append(
                {
                    "major_id": r.major_id,
                    "major_name": r.major_name,
                }
            )
        return majors

    def _get_all_majors_and_specialization_from_db(
        self, db: Session, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Lấy danh sách majors kèm theo danh sách specializations
        """
        rows = db.query(Major).order_by(Major.major_name).limit(limit).all()

        majors = []
        for r in rows:
            majors.append(
                {
                    "major_id": r.major_id,
                    "major_name": r.major_name,
                    "specializations": [
                        {
                            "specialization_id": s.specialization_id,
                            "specialization_name": s.specialization_name,
                        }
                        for s in r.specializations
                    ],
                }
            )

        return majors


langchain_service = TrainingService()
