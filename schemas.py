"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Example schemas (retain for reference):
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Trading-related schemas used by the app
class Signal(BaseModel):
    pair: str
    type: str  # LONG or SHORT
    price: float
    confidence: int
    size_usdt: float
    leverage: int
    entry_low: float
    entry_high: float
    tp1: float
    tp2: float
    tp3: float
    sl: float
    risk_usdt: float
    capital_example: float
    projected_roi_pct: float
    reason: str
    timestamp: int

class SignalPerformance(BaseModel):
    signal_id: str
    pair: str
    outcome: str  # hit_tp, hit_sl, breakeven, open
    roi_pct: float
    closed_at: Optional[int] = None
    notes: Optional[str] = None
