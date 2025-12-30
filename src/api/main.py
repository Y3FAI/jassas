"""
Jassas Search API - FastAPI application.
"""
import sys
import os
import time
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ranker.engine import Ranker
from api.schemas import SearchRequest, SearchResponse, SearchResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Ranker on startup, cleanup on shutdown."""
    print("🚀 Starting Jassas Search API...")
    try:
        app.state.ranker = Ranker(verbose=True)
        print("✅ Ranker loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Ranker: {e}")
        raise

    yield

    print("🛑 Shutting down Jassas Search API")


app = FastAPI(
    title="Jassas Search API",
    description="Hybrid search engine for Saudi government services",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_snippet(text: str, length: int = 200) -> str:
    """Generate snippet from text content."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[:length] + "..."


def execute_search(request: Request, query: str, limit: int) -> SearchResponse:
    """Core search logic shared by GET and POST endpoints."""
    start_time = time.perf_counter()

    ranker: Ranker = getattr(request.app.state, "ranker", None)
    if not ranker:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    try:
        raw_results = ranker.search(query=query, k=limit)

        formatted_results = []
        for res in raw_results:
            formatted_results.append(
                SearchResult(
                    title=res.get("title") or "No Title",
                    url=res.get("url") or "",
                    snippet=generate_snippet(res.get("clean_text", "")),
                    score=round(res.get("score", 0.0), 4)
                )
            )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            query=query,
            count=len(formatted_results),
            execution_time_ms=round(execution_time_ms, 2),
            results=formatted_results
        )

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")


@app.post("/api/v1/search", response_model=SearchResponse)
def search_post(request: Request, payload: SearchRequest):
    """Search endpoint (POST) - for programmatic use."""
    return execute_search(request, payload.query, payload.limit)


@app.get("/api/v1/search", response_model=SearchResponse)
def search_get(request: Request, q: str, limit: int = 10):
    """Search endpoint (GET) - browser friendly."""
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    return execute_search(request, q, min(limit, 50))


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    is_ready = hasattr(request.app.state, "ranker")
    return {
        "status": "online",
        "engine_ready": is_ready
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
