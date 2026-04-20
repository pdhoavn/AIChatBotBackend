from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from app.core.config import settings

from app.api.routes import (
    knowledge_base_controller,
    chat_controller,
    auth_controller,
    profile_controller,
    major_controller,
    specialization_controller,
    article_controller,
    users_controller,
    riasec_controller,
    permissions_controller,
    academic_score_controller,
    live_chat_controller,
    intent_controller,
    template_controller,
    analytics_controller
)
from app.models.database import init_db
import os

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    # Add security scheme for Swagger docs
    openapi_tags=[
        {"name": "Authentication", "description": "Authentication operations"},
        {"name": "Users", "description": "User management operations"},
        {"name": "Intent", "description": "Intent management operations"},
        {"name": "Knowledge Base", "description": "Knowledge base operations"},
        {"name": "Training Questions", "description": "Training Q&A operations"},
    ]
)

# Add security scheme to OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description="Admission Consulting Chatbot API with JWT Authentication",
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter JWT token obtained from /auth/login"
        }
    }
    
    # Add security requirement to all endpoints (except auth)
    for path, path_item in openapi_schema["paths"].items():
        if not path.startswith("/auth"):
            for operation in path_item.values():
                if isinstance(operation, dict) and "operationId" in operation:
                    operation["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler to ensure CORS headers are always present
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
    # Manually add CORS headers to error responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )
    # Manually add CORS headers to error responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*" 
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

async def startup_event():
    init_db()
app.add_event_handler("startup",startup_event)


app.include_router(live_chat_controller.router, prefix="/live_chat")
app.include_router(auth_controller.router, prefix="/auth", tags=["Authentication"])
app.include_router(users_controller.router, prefix="/users", tags=["Users"])
app.include_router(profile_controller.router, prefix="/profile", tags=["Profile"])
app.include_router(major_controller.router, prefix="/majors", tags=["Majors"])
app.include_router(specialization_controller.router, prefix="/specializations", tags=["Specializations"])
app.include_router(article_controller.router, prefix="/articles", tags=["Articles"])
app.include_router(knowledge_base_controller.router, prefix="/knowledge", tags=["Knowledge Base"])
app.include_router(chat_controller.router, prefix="/chat", tags=["Chat"])
app.include_router(riasec_controller.router, prefix="/riasec", tags=["RIASEC"])
app.include_router(permissions_controller.router, prefix="/permissions", tags=["Permissions"])
app.include_router(academic_score_controller.router, prefix="/academic-score", tags=["Academic Score"])
app.include_router(intent_controller.router, prefix="/intent", tags=["Intent"])
app.include_router(template_controller.router, prefix="/template", tags=["Template"])
app.include_router(analytics_controller.router, prefix="/analytics", tags=["Analytics"])

@app.get("/")
async def root():
    return {"message": "FastAPI + LangChain + Qdrant + OpenAI API"}



