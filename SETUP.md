# AIChatBotBackend — Local Development Setup Guide

> The database (PostgreSQL) and vector store (Qdrant) are hosted on a remote server.
> No local database installation is required.

---

## Requirements

| Tool        | Minimum Version  |
|-------------|------------------|
| Python      | 3.12+            |
| pip / venv  | Bundled with Python |
| Git         | Any              |

---

## 1. Clone the repository

```bash
git clone <REPO_URL> AIChatBotBackend
cd AIChatBotBackend
```

---

## 2. Create and activate a virtual environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Align qdrant-client version with the hosted server (1.17.1)
pip install "qdrant-client==1.17.1"
```

> The `requirements.txt` pins `qdrant-client==1.15.1`, but the hosted server
> runs version `1.17.1`. Installing the matching version eliminates the
> incompatibility warning at startup.

---

## 4. Configure the `.env` file

Create a `.env` file at the project root with the following variables.
All connection strings must point to the hosted remote server.

```env
# ===== API Keys =====
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# ===== Database (remote hosted server) =====
# Ensure POSTGRES_USER and POSTGRES_DB exactly match the credentials in DATABASE_URL.
DATABASE_URL=postgresql://<USER>:<PASSWORD>@<SERVER_IP>:5432/<DB_NAME>

POSTGRES_USER=<USER>
POSTGRES_PASSWORD=<PASSWORD>
POSTGRES_DB=<DB_NAME>

# ===== Qdrant vector store (remote hosted server) =====
QDRANT_HOST=<SERVER_IP>
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents

# ===== AI Models =====
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4

# ===== JWT =====
SECRET_KEY=<min_32_character_random_string>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ===== Cloudinary =====
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
```

> Obtain `<SERVER_IP>`, `<USER>`, `<PASSWORD>`, and `<DB_NAME>` from the
> infrastructure owner or team lead.

---

## 5. Run the development server

```bash
python run.py
```

The server starts at `http://localhost:8000` with hot-reload enabled.

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...] using WatchFiles
INFO:     Application startup complete.
```

---

## 6. Verify the server is running

| URL                             | Description                        |
|---------------------------------|------------------------------------|
| `http://localhost:8000/docs`    | Swagger UI — interactive API docs  |
| `http://localhost:8000/redoc`   | ReDoc — alternative API reference  |

---

## Known Warnings (non-blocking)

### Pydantic V2 config key rename

```
UserWarning: 'orm_mode' has been renamed to 'from_attributes'
```

This is a deprecation warning from the Pydantic V2 migration. The application
runs normally. No action is required.

### Qdrant client/server version mismatch

```
UserWarning: Qdrant client version X.X.X is incompatible with server version Y.Y.Y
```

Re-run the pinned install from Step 3 to resolve:

```bash
pip install "qdrant-client==1.17.1"
```

---

## Code Changes Applied to This Repository

The following changes were made to enable the application to connect to the
hosted remote database without issues.

### `app/main.py` — Startup event handler

**Problem:** `Base.metadata.create_all()` is a synchronous blocking call. When
invoked inside an `async` startup handler, it blocks the Uvicorn event loop.
With a remote database, each of the ~25 table existence checks requires a
separate TCP round-trip, collectively exceeding the default socket timeout.

**Fix:** `create_all()` was removed from the startup handler. Because the
database schema is managed externally on the hosted server, table creation at
application startup is unnecessary. A lightweight `SELECT 1` ping replaces it
to confirm connectivity before the application begins accepting requests.

```python
async def startup_event() -> None:
    """
    Application startup handler.

    The database schema is managed externally on the hosted server.
    Table creation via create_all() is intentionally omitted here to avoid
    blocking the async event loop during startup (create_all is synchronous
    and performs one round-trip per table, which causes timeouts on remote
    connections).

    Instead, a lightweight connectivity check is performed to confirm that
    the application can reach the database before accepting requests.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connectivity check passed.")
    except Exception as exc:
        logger.error("Database connectivity check failed: %s", exc)
```

### `app/models/database.py` — SQLAlchemy engine configuration

**Problem:** The engine was created with no connection timeout, pool size, or
pre-ping settings, making it vulnerable to stale / dropped connections on a
remote host.

**Fix:** Added explicit connection arguments and pool settings.

```python
engine = create_engine(
    DATABASE_URL,
    connect_args={"connect_timeout": 10},
    pool_size=2,
    max_overflow=3,
    pool_timeout=30,
    pool_pre_ping=True,
)
```

---

## Project Structure

```
AIChatBotBackend/
├── app/
│   ├── api/          # Route handlers (controllers)
│   ├── core/         # Application config and security utilities
│   ├── models/       # SQLAlchemy ORM models and database session
│   ├── services/     # Business logic layer
│   ├── utils/        # Shared utility functions
│   └── main.py       # FastAPI application entry point
├── requirements.txt
├── run.py            # Development server entry point
├── .env              # Environment variables (create manually, do not commit)
└── SETUP.md          # This file
```

---

## Stop the server

Press `Ctrl + C` in the terminal running the server.

---

## Docker (production deployment)

Docker is not required for local development. For production deployment, refer
to `docker-compose.yaml`. Ensure all environment variables in `.env` are
correct before building:

```bash
docker compose up --build
```
