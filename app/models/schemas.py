from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date


# ================= AUTH =================
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[int] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserProfileResponse(BaseModel):
    user_id: int
    full_name: str 
    email: str
    phone_number: Optional[str]
    permission: Optional[List[str]]
    role_name: Optional[str]
    student_profile: Optional[dict] = None
    consultant_profile: Optional[dict] = None
    content_manager_profile: Optional[dict] = None
    admission_official_profile: Optional[dict] = None
    # Add explicit leadership flags for frontend
    consultant_is_leader: Optional[bool] = False
    content_manager_is_leader: Optional[bool] = False

    class Config:
        orm_mode = True


# ================= ROLE =================
class RoleBase(BaseModel):
    role_name: str


class RoleCreate(RoleBase):
    pass


class RoleResponse(RoleBase):
    role_id: int

    class Config:
        orm_mode = True


# ================= USER =================
class UserBase(BaseModel):
    full_name: str
    email: EmailStr
    status: Optional[bool] = None


class UserCreate(UserBase):
    password: str
    role_id: Optional[int] = None
    permissions: Optional[List[int]] = None  # List of permission IDs
    phone_number: Optional[str] = None
    # Optional flags to set when creating profiles for certain permissions
    consultant_is_leader: Optional[bool] = False
    content_manager_is_leader: Optional[bool] = False
    # Optional interest information to create CustomerProfile at registration
    interest_desired_major: Optional[str] = None
    interest_region: Optional[str] = None


class PermissionChangeRequest(BaseModel):
    user_id: int
    permission_ids: List[int]
    consultant_is_leader: Optional[bool] = False
    content_manager_is_leader: Optional[bool] = False


class PermissionRevokeRequest(BaseModel):
    user_id: int
    permission_ids: List[int]
class BanUserRequest(BaseModel):
    user_id: int


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None
    status: Optional[bool] = None


class UserResponse(UserBase):
    user_id: int
    role_id: Optional[int]
    permissions: Optional[List[int]] = []  # List of permission IDs

    class Config:
        orm_mode = True


# ================= STUDENT PROFILE =================
class StudentProfileBase(BaseModel):
    interest_id: Optional[int]


class StudentProfileCreate(StudentProfileBase):
    pass


class StudentProfileResponse(StudentProfileBase):
    student_id: int

    class Config:
        orm_mode = True


# ================= INTEREST =================
class InterestBase(BaseModel):
    desired_major: Optional[str]
    region: Optional[str]


class InterestCreate(InterestBase):
    pass


class InterestResponse(InterestBase):
    interest_id: int

    class Config:
        orm_mode = True


# ================= ACADEMIC SCORE =================
class AcademicScoreBase(BaseModel):
    math: float
    literature: float
    english: float
    physics: float
    chemistry: float
    biology: float
    history: float
    geography: float


class AcademicScoreCreate(AcademicScoreBase):
    pass


class AcademicScoreResponse(AcademicScoreBase):
    score_id: int
    customer_id: int
    
    class Config:
        orm_mode = True


# ================= CONSULTANT PROFILE =================
class ConsultantProfileBase(BaseModel):
    rating: Optional[int]
    status: Optional[str]
    is_leader: Optional[bool]


class ConsultantProfileResponse(ConsultantProfileBase):
    consultant_id: int

    class Config:
        orm_mode = True


# ================= CONTENT MANAGER PROFILE =================
class ContentManagerProfileBase(BaseModel):
    is_leader: bool


class ContentManagerProfileResponse(ContentManagerProfileBase):
    content_manager_id: int

    class Config:
        orm_mode = True


# ================= ADMISSION OFFICIAL PROFILE =================
class AdmissionOfficialProfileBase(BaseModel):
    rating: Optional[int]
    current_sessions: Optional[int]
    max_sessions: Optional[int]
    status: Optional[str]


class AdmissionOfficialProfileResponse(AdmissionOfficialProfileBase):
    admission_official_id: int

    class Config:
        orm_mode = True


# ================= CURRICULUM / MAJOR / COURSE =================
class CourseBase(BaseModel):
    name: str
    description: Optional[str]
    semester: Optional[str]
    major_id: Optional[int]


class CourseResponse(CourseBase):
    course_id: int

    class Config:
        orm_mode = True


class MajorBase(BaseModel):
    major_name: str

class MajorResponse(MajorBase):
    major_id: int

    class Config:
        orm_mode = True


class SpecializationBase(BaseModel):
    specialization_id: int
    specialization_name: str

    class Config:
        orm_mode = True


class ArticleBase(BaseModel):
    article_id: int
    title: str
    description: Optional[str]
    url: Optional[str]
    create_at: Optional[date]
    specialization: Optional[SpecializationBase]

    class Config:
        orm_mode = True


class AdmissionFormBase(BaseModel):
    form_id: int
    fullname: str
    email: str
    phone_number: Optional[str]
    campus: Optional[str]
    submit_time: Optional[date]

    class Config:
        orm_mode = True


class MajorDetailResponse(MajorResponse):
    articles: List[ArticleBase] = []
    admission_forms: List[AdmissionFormBase] = []

    class Config:
        orm_mode = True


class CurriculumBase(BaseModel):
    curriculum_name: str
    description: Optional[str]
    tuition_fee: Optional[float]
    image: Optional[str]


class CurriculumResponse(CurriculumBase):
    curriculum_id: int
    majors: Optional[List[MajorResponse]] = []

    class Config:
        orm_mode = True


# ================= INTENT / TRAINING QUESTION / FAQ =================
class IntentBase(BaseModel):
    intent_name: str
    description: Optional[str]
    is_deleted: Optional[bool] = False


class IntentResponse(IntentBase):
    intent_id: int

    class Config:
        orm_mode = True


class TrainingQuestionRequest(BaseModel):
    question: str
    answer: str
    intent_id: Optional[int]


class TrainingQuestionResponse(TrainingQuestionRequest):
    question_id: int
    intent_name: Optional[str]
    status: Optional[str] = "draft"  # draft, approved, rejected, deleted
    created_at: Optional[date] = None
    approved_at: Optional[date] = None
    created_by: Optional[int] = None
    approved_by: Optional[int] = None
    reject_reason: Optional[str] = None

    class Config:
        orm_mode = True


class FaqStatisticsBase(BaseModel):
    usage_count: int
    success_rate: float
    question_text: str
    last_used_at: Optional[date]
    intent_id: Optional[int]


class FaqStatisticsResponse(FaqStatisticsBase):
    faq_id: int

    class Config:
        orm_mode = True


# ================= KNOWLEDGE BASE =================
class KnowledgeBaseDocumentBase(BaseModel):
    title: str
    file_path: str
    category: Optional[str]
    created_at: Optional[date]
    updated_at: Optional[date]
    created_by: Optional[int]
    reject_reason: Optional[str] = None


class KnowledgeBaseDocumentResponse(KnowledgeBaseDocumentBase):
    document_id: int
    status: Optional[str] = "draft"  # draft, approved, rejected, deleted
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[date] = None
    reject_reason: Optional[str] = None

    class Config:
        orm_mode = True


class DocumentChunkBase(BaseModel):
    chunk_text: str
    created_at: Optional[date]
    document_id: int


class DocumentChunkResponse(DocumentChunkBase):
    chunk_id: int

    class Config:
        orm_mode = True


# ================= CHAT =================
class ChatInteractionBase(BaseModel):
    message_text: str
    timestamp: Optional[date]
    is_from_bot: bool
    sender_id: Optional[int]
    session_id: Optional[int]


class ChatInteractionResponse(ChatInteractionBase):
    interaction_id: int

    class Config:
        orm_mode = True


class ChatSessionBase(BaseModel):
    session_type: str
    start_time: Optional[date]
    end_time: Optional[date]
    feedback_rating: Optional[int]
    notes: Optional[str]
    student_id: Optional[int]
    admission_officer_id: Optional[int]


class ChatSessionResponse(ChatSessionBase):
    chat_session_id: int
    interactions: Optional[List[ChatInteractionResponse]] = []

    class Config:
        orm_mode = True


# ================= SPECIALIZATION =================
class SpecializationResponse(BaseModel):
    specialization_id: int
    specialization_name: str
    major_id: Optional[int]
    articles: List['ArticleResponse'] = []

    class Config:
        orm_mode = True

# ================= ARTICLE =================


class ArticleCreate(BaseModel):
    title: str
    description: str
    url: Optional[str] = None
    link_image: Optional[str] = None
    note: Optional[str] = None
    major_id: Optional[int] = None
    specialization_id: Optional[int] = None

class ArticleUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    link_image: Optional[str] = None
    note: Optional[str] = None
    major_id: Optional[int] = None
    specialization_id: Optional[int] = None

class ArticleStatusUpdate(BaseModel):
    status: str  # "draft", "published", "rejected" or "cancelled"
    note: Optional[str] = None

class ArticleResponse(BaseModel):
    article_id: int
    title: str
    description: str
    url: Optional[str]
    link_image: Optional[str] = None
    note: Optional[str] = None
    status: str
    create_at: date
    created_by: int
    major_id: Optional[int] = None
    specialization_id: Optional[int] = None
    author_name: Optional[str] = None
    major_name: Optional[str] = None
    specialization_name: Optional[str] = None
    note: Optional[str] = None

    class Config:
        orm_mode = True


# ================= RECOMMENDATION =================
class PersonalizedRecommendationBase(BaseModel):
    confidence_score: float
    user_id: int
    base_intent_id: int
    suggested_intent_id: int
    session_id: int


class PersonalizedRecommendationResponse(PersonalizedRecommendationBase):
    recommendation_id: int

    class Config:
        orm_mode = True


# ================= RIASEC =================
class RiasecResultBase(BaseModel):
    score_realistic: int
    score_investigative: int
    score_artistic: int
    score_social: int
    score_enterprising: int
    score_conventional: int
    result: str

class RiasecResultCreate(RiasecResultBase):
    pass

class RiasecResult(RiasecResultBase):
    result_id: int
    customer_id: Optional[int] = None

    class Config:
        orm_mode = True

# ================= TEMPLATE =================
class TemplateQABase(BaseModel):
    question: str
    answer: str
    order_position: int = 0

class TemplateQACreate(TemplateQABase):
    pass

class TemplateQAUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    order_position: Optional[int] = None

class TemplateQAResponse(TemplateQABase):
    qa_id: int
    template_id: int

    class Config:
        orm_mode = True


class TemplateBase(BaseModel):
    template_name: str
    description: Optional[str] = None

class TemplateCreate(TemplateBase):
    qa_pairs: List[TemplateQACreate]

class TemplateUpdate(BaseModel):
    template_name: Optional[str] = None
    description: Optional[str] = None
    qa_pairs: Optional[List[TemplateQAUpdate]] = None

class TemplateDelete(BaseModel):
    template_ids: List[int]


class TemplateResponse(TemplateBase):
    template_id: int
    is_active: bool
    created_by: Optional[int] = None
    qa_pairs: List[TemplateQAResponse] = []

    class Config:
        orm_mode = True
