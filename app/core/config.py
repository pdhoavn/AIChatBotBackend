from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI LangChain App"
    VERSION: str = "1.0.0"

    OPENAI_API_KEY: str
    GEMINI_API_KEY: str
    DATABASE_URL: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    QDRANT_HOST: str
    QDRANT_PORT: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str

    EMBEDDING_MODEL: str
    LLM_MODEL: str
    CHAT_UNIVERSITY_NAME: str
    TOP_K: int
    CONFIDENCE_SCORE: float
    CROSS_ENCODER_SCORE: float
    # JWT
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

    # OCR / Tesseract
    TESSERACT_CMD_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Facebook Messenger
    FACEBOOK_PAGE_ACCESS_TOKEN: str
    FACEBOOK_VERIFY_TOKEN: str = "messenger_webhook_verify_token"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "forbid"  # Có thể đổi thành "ignore" nếu muốn bỏ qua biến dư


settings = Settings()
