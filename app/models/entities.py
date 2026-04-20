import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Date, Float, ForeignKey, Text
)
from datetime import datetime
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine

Base = declarative_base()
# =====================
# USERS, ROLE, PERMISSION
# =====================
class Users(Base):
    __tablename__ = 'Users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    status = Column(Boolean, default=True)
    role_id = Column(Integer, ForeignKey('Role.role_id'), nullable=True)
    phone_number = Column(String, nullable=False)

    # Relationships
    role = relationship('Role', back_populates='users')

    # permissions (many-to-many)
    user_permissions = relationship('UserPermission', back_populates='user', cascade="all, delete-orphan")
    permissions = relationship('Permission', secondary='UserPermission', back_populates='users', overlaps="user,user_permissions")

    # 1-1 profiles
    customer_profile = relationship('CustomerProfile', back_populates='user', uselist=False)
    consultant_profile = relationship('ConsultantProfile', back_populates='user', uselist=False)
    content_manager_profile = relationship('ContentManagerProfile', back_populates='user', uselist=False)
    admission_official_profile = relationship('AdmissionOfficialProfile', back_populates='user', uselist=False)

    # documents / knowledge base
    knowledge_documents = relationship(
        'KnowledgeBaseDocument', 
        foreign_keys='[KnowledgeBaseDocument.created_by]',
        back_populates='author', 
        cascade="all, delete-orphan"
    )
    reviewed_knowledge_documents = relationship(
        'KnowledgeBaseDocument',
        foreign_keys='[KnowledgeBaseDocument.reviewed_by]',
        back_populates='reviewer'
    )
    document_chunks = relationship('DocumentChunk', back_populates='created_by_user', cascade="all, delete-orphan")

    # templates & articles & admissions
    templates = relationship('Template', back_populates='creator', cascade="all, delete-orphan")
    articles = relationship('Article', back_populates='author_user', cascade="all, delete-orphan")
    admission_informations = relationship('AdmissionInformation', back_populates='creator', cascade="all, delete-orphan")
    admission_forms = relationship('AdmissionForm', back_populates='user', cascade="all, delete-orphan")

    # chat, recommendations
    chat_interactions = relationship('ChatInteraction', back_populates='user', cascade="all, delete-orphan")
    participate_sessions = relationship('ParticipateChatSession', back_populates='user')
    # training QA created/approved/rejected (three distinct relations)
    training_question_answers_created = relationship(
        "TrainingQuestionAnswer",
        foreign_keys="[TrainingQuestionAnswer.created_by]",
        back_populates="created_by_user",
        cascade="all, delete-orphan"
    )
    training_question_answers_approved = relationship(
        "TrainingQuestionAnswer",
        foreign_keys="[TrainingQuestionAnswer.approved_by]",
        back_populates="approved_by_user",
        cascade="all, delete-orphan"
    )
    # Note: rejected_by link removed; rejection is stored as text in TrainingQuestionAnswer.reject_reason


class UserPermission(Base):
    __tablename__ = 'UserPermission'
    
    permission_id = Column(Integer, ForeignKey("Permission.permission_id"), primary_key=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    
    # Relationships
    user = relationship('Users', back_populates='user_permissions')
    permission = relationship('Permission', back_populates='user_permissions')


class Permission(Base):
    __tablename__ = 'Permission'
    
    permission_id = Column(Integer, primary_key=True, autoincrement=True)
    permission_name = Column(String)
    
    # Relationships
    user_permissions = relationship('UserPermission', back_populates='permission', cascade="all, delete-orphan")
    users = relationship('Users', secondary='UserPermission', back_populates='permissions', overlaps="user,user_permissions")


class Role(Base):
    __tablename__ = 'Role'
    
    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String)
    
    # Relationships
    users = relationship('Users', back_populates='role')


# =====================
# CUSTOMER PROFILE (was StudentProfile)
# =====================
class CustomerProfile(Base):
    __tablename__ = 'CustomerProfile'
    
    customer_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    interest_id = Column(Integer, ForeignKey('Interest.interest_id'), nullable=True)
    # Relationships
    user = relationship('Users', back_populates='customer_profile')
    interest = relationship('Interest', back_populates='customer_profiles')
    academic_scores = relationship('AcademicScore', back_populates='customer', cascade="all, delete-orphan")
    riasec_results = relationship('RiasecResult', back_populates='customer', cascade="all, delete-orphan")
   
class LiveChatQueue(Base):
    __tablename__ = 'LiveChatQueue'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("Users.user_id"))
    status = Column(String, default="waiting")  # waiting, accepted, canceled
    created_at = Column(Date, default=datetime.now)
    
    # relationships
    customer = relationship("Users", foreign_keys=[customer_id])
    

class Interest(Base):
    __tablename__ = 'Interest'
    
    interest_id = Column(Integer, primary_key=True, autoincrement=True)
    desired_major = Column(String)
    region = Column(String)
    
    # Relationships
    customer_profiles = relationship('CustomerProfile', back_populates='interest', cascade="all, delete-orphan")


class AcademicScore(Base):
    __tablename__ = 'AcademicScore'
    
    score_id = Column(Integer, primary_key=True, autoincrement=True)
    math = Column(Float)
    literature = Column(Float)
    english = Column(Float)
    physics = Column(Float)
    chemistry = Column(Float)
    biology = Column(Float)
    history = Column(Float)
    geography = Column(Float)

    customer_id = Column(Integer, ForeignKey('CustomerProfile.customer_id'))
    
    # Relationships
    customer = relationship('CustomerProfile', back_populates='academic_scores')


# =====================
# RIASEC
# =====================


class RiasecResult(Base):
    __tablename__ = 'RiasecResult'
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    score_realistic = Column(Integer)
    score_investigative = Column(Integer)
    score_artistic = Column(Integer)
    score_social = Column(Integer)
    score_enterprising = Column(Integer)
    score_conventional = Column(Integer)
    result = Column(String)
    # session_id = Column(String, unique=True)
    customer_id = Column(Integer, ForeignKey('CustomerProfile.customer_id'))
    
    # Relationships
    customer = relationship('CustomerProfile', back_populates='riasec_results')


# =====================
# CONSULTANT, MANAGER, ADMISSION OFFICIAL
# =====================
class ConsultantProfile(Base):
    __tablename__ = 'ConsultantProfile'
    
    consultant_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    status = Column(Boolean, default=True)
    is_leader = Column(Boolean, default=False)
   
    user = relationship('Users', back_populates='consultant_profile')


class ContentManagerProfile(Base):
    __tablename__ = 'ContentManagerProfile'
    
    content_manager_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    is_leader = Column(Boolean, default=False)
  
    user = relationship('Users', back_populates='content_manager_profile')


class AdmissionOfficialProfile(Base):
    __tablename__ = 'AdmissionOfficialProfile'
    
    admission_official_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    rating = Column(Integer)
    current_sessions = Column(Integer)
    max_sessions = Column(Integer)
    status = Column(String)
    
    user = relationship('Users', back_populates='admission_official_profile')


# =====================
# MAJOR, ADMISSION FORM
# =====================
class Major(Base):
    __tablename__ = 'Major'
    
    major_id = Column(Integer, primary_key=True, autoincrement=True)
    major_name = Column(String, nullable=False)
    created_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)
    
    admission_forms = relationship('AdmissionForm', back_populates='major', cascade="all, delete-orphan")
    articles = relationship('Article', back_populates='major', cascade="all, delete-orphan")
    specializations = relationship(
    'Specialization',
    back_populates='major',
    cascade="all, delete-orphan")
            

class AdmissionForm(Base):
    __tablename__ = 'AdmissionForm'
    
    form_id = Column(Integer, primary_key=True, autoincrement=True)
    fullname = Column(String)
    email = Column(String)
    phone_number = Column(String)
    major_id = Column(Integer, ForeignKey('Major.major_id'))
    campus = Column(String)
    submit_time = Column(Date)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    user = relationship('Users', back_populates='admission_forms')
    major = relationship('Major', back_populates='admission_forms')


# =====================
# CHAT SYSTEM
# =====================
class ChatSession(Base):
    __tablename__ = 'ChatSession'
    
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    session_type = Column(String)
    start_time = Column(Date, default=datetime.now)
    end_time = Column(Date)
    feedback_rating = Column(Integer)
    notes = Column(String)
   
    interactions = relationship('ChatInteraction', back_populates='session', passive_deletes=True)
    participate_sessions = relationship('ParticipateChatSession', back_populates='session', cascade="all, delete-orphan")


class ParticipateChatSession(Base):
    __tablename__ = 'ParticipateChatSession'
    
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    session_id = Column(Integer, ForeignKey('ChatSession.chat_session_id'), primary_key=True)
    
    # Relationships
    session = relationship('ChatSession', back_populates='participate_sessions')
    user = relationship('Users', back_populates='participate_sessions')  


class ChatInteraction(Base):
    __tablename__ = 'ChatInteraction'
    
    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    message_text = Column(Text)
    timestamp = Column(Date, default=datetime.now)
    rating = Column(Integer)
    is_from_bot = Column(Boolean)
    sender_id = Column(Integer, ForeignKey('Users.user_id'))
    session_id = Column(Integer, ForeignKey('ChatSession.chat_session_id', ondelete="SET NULL"), nullable=True)
    
    # Relationships
    user = relationship('Users', back_populates='chat_interactions')
    session = relationship('ChatSession', back_populates='interactions')
    faq_responses = relationship(
    'FaqStatistics',
    foreign_keys='FaqStatistics.response_from_chat_id',
    back_populates='response_from_chat'
    )

    faq_queries = relationship(
        'FaqStatistics',
        foreign_keys='FaqStatistics.query_from_user_id',
        back_populates='query_from_user'
    )

# =====================
# INTENT, FAQ, RECOMMENDATION, TRAINING QA
# =====================
class Intent(Base):
    __tablename__ = 'Intent'
    
    intent_id = Column(Integer, primary_key=True, autoincrement=True)
    intent_name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(Date, default=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    is_deleted = Column(Boolean, default=False)
    
    faq_statistics = relationship('FaqStatistics', back_populates='intent', cascade="all, delete-orphan")
    training_questions = relationship('TrainingQuestionAnswer', back_populates='intent', cascade="all, delete-orphan")
    document = relationship('KnowledgeBaseDocument', back_populates='intent', cascade="all, delete-orphan")

class FaqStatistics(Base):
    __tablename__ = 'FaqStatistics'
    
    faq_id = Column(Integer, primary_key=True, autoincrement=True)  
    response_from_chat_id = Column(Integer, ForeignKey(ChatInteraction.interaction_id))
    query_from_user_id = Column(Integer, ForeignKey(ChatInteraction.interaction_id))
    last_used_at = Column(Date)
    intent_id = Column(Integer, ForeignKey('Intent.intent_id'))
    usage_count = Column(Integer)
    response_from_chat = relationship(
    'ChatInteraction',
    foreign_keys=[response_from_chat_id],
    back_populates='faq_responses'
    )

    query_from_user = relationship(
        'ChatInteraction',
        foreign_keys=[query_from_user_id],
        back_populates='faq_queries'
    )
    intent = relationship('Intent', back_populates='faq_statistics')




class TrainingQuestionAnswer(Base):
    __tablename__ = 'TrainingQuestionAnswer'
    
    question_id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String)
    answer = Column(String)
    status = Column(String, default="draft")  # Values: draft, approved, rejected, deleted
    intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    created_at = Column(Date, default=datetime.now, nullable=True)
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    approved_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    approved_at = Column(Date, nullable=True)
    reject_reason = Column(String, nullable=True)
    # removed rejected_by/rejected_at: rejection author/date are not stored as separate columns
    
    # Relationships
    intent = relationship("Intent", back_populates="training_questions")

    created_by_user = relationship(
        "Users", foreign_keys=[created_by], back_populates="training_question_answers_created"
    )
    approved_by_user = relationship(
        "Users", foreign_keys=[approved_by], back_populates="training_question_answers_approved"
    )
    # rejection author relationship removed


# -------------------- AdmissionInformation ---------------------------
class AdmissionInformation(Base):
    __tablename__ = "AdmissionInformation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    academic_year = Column(String)
    target_applicant = Column(String)
    admission_method = Column(String)
    scholarship_infor = Column(String)
    create_at = Column(Date, default=datetime.now)
    update_at = Column(Date, onupdate=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"))

    # Relationships
    creator = relationship("Users", back_populates="admission_informations")
# ---------------------------------------------------------------------


# --------------- KnowledgeBaseDocument & DocumentChunk ----------------
class KnowledgeBaseDocument(Base):
    __tablename__ = 'KnowledgeBaseDocument'
    
    document_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    file_path = Column(String)
    category = Column(String)
    intend_id = Column(Integer, ForeignKey('Intent.intent_id'))
    status = Column(String, default="draft")  # Values: draft, approved, rejected, deleted
    created_at = Column(Date, default=datetime.now)
    updated_at = Column(Date, onupdate=datetime.now)
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    reviewed_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)
    reviewed_at = Column(Date, nullable=True)
    reject_reason = Column(String, nullable=True)
    
    intent = relationship('Intent', back_populates='document')
    # Relationships
    chunks = relationship('DocumentChunk', back_populates='document', cascade="all, delete-orphan")
    author = relationship('Users', foreign_keys=[created_by], back_populates='knowledge_documents')
    reviewer = relationship('Users', foreign_keys=[reviewed_by], back_populates='reviewed_knowledge_documents')
    

class DocumentChunk(Base):
    __tablename__ = 'DocumentChunk'
    
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text)
    embedding_vector = Column(String)  # Store as JSON or use vector extension
    created_at = Column(Date, default=datetime.now)
    document_id = Column(Integer, ForeignKey('KnowledgeBaseDocument.document_id'))
    created_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)

    # Relationships
    document = relationship('KnowledgeBaseDocument', back_populates='chunks')
    created_by_user = relationship('Users', back_populates='document_chunks')
# ---------------------------------------------------------------------


# -------------------- Template & Template_QA ----------------------
class Template(Base):
    __tablename__ = 'Template'
    
    template_id = Column(Integer, primary_key=True, autoincrement=True)
    template_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    qa_pairs = relationship('Template_QA', back_populates='template', cascade="all, delete-orphan")
    creator = relationship('Users', back_populates='templates')


class Template_QA(Base):
    """Template Q&A pairs for consultants to use as examples when creating training questions"""
    __tablename__ = 'Template_QA'
    
    qa_id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("Template.template_id"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    order_position = Column(Integer, default=0)
    
    # Relationships
    template = relationship('Template', back_populates='qa_pairs')
# ---------------------------------------------------------------------


# ---------------- Specialization & Article ---------------------------
class Specialization(Base):
    __tablename__ = 'Specialization'
    
    specialization_id = Column(Integer, primary_key=True, autoincrement=True)
    specialization_name = Column(String, nullable=False)
    major_id = Column(Integer, ForeignKey('Major.major_id'), nullable=True)
    
    articles = relationship('Article', back_populates='specialization', cascade="all, delete-orphan")
    major = relationship('Major', back_populates='specializations')

class Article(Base):
    __tablename__ = "Article"

    article_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    description = Column(String)
    url = Column(String)
    link_image = Column(String)
    note = Column(String)
    # content = Column(Text)
    status = Column(String, default="draft")  # Values: draft, published, rejected, cancelled
    create_at = Column(Date, default=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    major_id = Column(Integer, ForeignKey('Major.major_id'), nullable=True)
    specialization_id = Column(Integer, ForeignKey('Specialization.specialization_id'), nullable=True)

    # Relationships
    author_user = relationship('Users', back_populates='articles')
    major = relationship('Major', back_populates='articles')
    specialization = relationship('Specialization', back_populates='articles')
