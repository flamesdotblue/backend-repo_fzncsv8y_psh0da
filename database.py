import os
from typing import Any, Dict, List, Optional

# Lazy/optional Motor import to avoid hard dependency at startup
try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase  # type: ignore
except Exception:  # ImportError or version mismatch
    AsyncIOMotorClient = None  # type: ignore
    AsyncIOMotorDatabase = None  # type: ignore

DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "alphadesk")

_client = None
_db = None

async def get_db():
    global _client, _db
    if AsyncIOMotorClient is None:
        raise RuntimeError("MongoDB driver not available")
    if _db is None:
        _client = AsyncIOMotorClient(DATABASE_URL)
        _db = _client[DATABASE_NAME]
    return _db

async def create_document(collection_name: str, data: Dict[str, Any]) -> str:
    try:
        db = await get_db()
        res = await db[collection_name].insert_one({**data, "_created_at": data.get("_created_at")})
        return str(res.inserted_id)
    except Exception:
        # Silently no-op if DB not available
        return ""

async def get_documents(collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        db = await get_db()
        cur = db[collection_name].find(filter_dict or {}).sort("_id", -1).limit(limit)
        return [doc async for doc in cur]
    except Exception:
        return []

async def ping() -> bool:
    try:
        db = await get_db()
        await db.command("ping")
        return True
    except Exception:
        return False
