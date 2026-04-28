from typing import Any, Dict, List, Optional
from functools import lru_cache

from qdrant_client import QdrantClient, models
from pydantic_settings import BaseSettings


class QdrantConfig(BaseSettings):
    """MCP-style configuration for Qdrant vector store."""
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_USE_HTTPS: bool = False
    QDRANT_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_qdrant_config() -> QdrantConfig:
    return QdrantConfig()


class QdrantManager:
    """Singleton MCP client for Qdrant operations.
    
    Encapsulates all Qdrant interactions to keep controllers/services clean.
    """
    _instance: Optional["QdrantManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[QdrantConfig] = None):
        if self._initialized:
            return
        self.config = config or get_qdrant_config()
        self.client = QdrantClient(
            host=self.config.QDRANT_HOST,
            port=self.config.QDRANT_PORT,
            api_key=self.config.QDRANT_API_KEY or None,
            https=self.config.QDRANT_USE_HTTPS,
            timeout=self.config.QDRANT_TIMEOUT,
            check_compatibility=False,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        info = self.client.get_collection(collection_name)
        return {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
        }

    def list_collections(self) -> List[str]:
        collections = self.client.get_collections()
        return [c.name for c in collections.collections]

    # ------------------------------------------------------------------
    # Points / Chunks
    # ------------------------------------------------------------------
    def count_points(self, collection_name: str, document_id: int) -> int:
        result = self.client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        return result.count

    def scroll_document_chunks(
        self,
        collection_name: str,
        document_id: int,
        limit: int = 1000,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        points, _next_page = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            {
                "point_id": point.id,
                "chunk_index": point.payload.get("chunk_index"),
                "chunk_text": point.payload.get("chunk_text"),
                "metadata": point.payload.get("metadata"),
            }
            for point in points
        ]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False


def get_qdrant_client() -> QdrantManager:
    return QdrantManager()
