from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Signal(BaseModel):
    pair: str
    type: str  # LONG or SHORT
    price: float
    confidence: int = Field(ge=0, le=100)
    entry_low: float
    entry_high: float
    tp1: float
    tp2: float
    tp3: float
    sl: float
    size_usdt: float
    leverage: int
    risk_usdt: float
    projected_roi_pct: float
    reason: str
    timestamp: datetime

class SignalPerformance(BaseModel):
    pair: str
    signal_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    pnl_usdt: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    rr: Optional[float] = None
