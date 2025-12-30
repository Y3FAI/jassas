"""
Pydantic schemas for Jassas Search API.
"""
from pydantic import BaseModel, Field
from typing import List


class SearchRequest(BaseModel):
    """Search request payload."""
    query: str = Field(..., min_length=2, example="الأمن السيبراني")
    limit: int = Field(default=10, ge=1, le=50, example=10)


class SearchResult(BaseModel):
    """Single search result."""
    title: str
    url: str
    snippet: str = Field(default="", description="First 200 chars of content")
    score: float


class SearchResponse(BaseModel):
    """Search response with results and metadata."""
    query: str
    count: int
    execution_time_ms: float
    results: List[SearchResult]
