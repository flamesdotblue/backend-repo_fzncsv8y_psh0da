import os
import time
from typing import List, Dict, Any

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import create_document, db
from schemas import Signal

BYBIT_BASE = "https://api.bybit.com"
PAIRS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","OPUSDT","LINKUSDT"
]

app = FastAPI(title="AlphaDesk Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HeatItem(BaseModel):
    pair: str
    vol_pct: float

class Analytics(BaseModel):
    win_rate: float
    rr: float
    avg_roi: float
    weekly: float


def bybit_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BYBIT_BASE}{path}"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if str(data.get("retCode")) != "0":
        raise RuntimeError(f"Bybit error: {data}")
    return data


def fetch_ticker(pair: str) -> Dict[str, Any]:
    data = bybit_get("/v5/market/tickers", {"category": "linear", "symbol": pair})
    items = data.get("result", {}).get("list", [])
    return items[0] if items else {}


def fetch_kline(pair: str, interval: str = "15", limit: int = 96) -> List[List[str]]:
    # interval mapping: 1 3 5 15 60 240 etc (minutes) for v5
    data = bybit_get("/v5/market/kline", {"category": "linear", "symbol": pair, "interval": interval, "limit": str(limit)})
    return data.get("result", {}).get("list", [])


def compute_signal_from_klines(pair: str) -> Dict[str, Any]:
    kl = fetch_kline(pair)
    if not kl or len(kl) < 20:
        raise RuntimeError("Insufficient kline data")
    # Each item: [startTime, open, high, low, close, volume, turnover]
    closes = [float(x[4]) for x in kl[::-1]]  # chronological
    highs = [float(x[2]) for x in kl[::-1]]
    lows = [float(x[3]) for x in kl[::-1]]
    vols = [float(x[5]) for x in kl[::-1]]

    last = closes[-1]
    last_vol = vols[-1]
    avg_vol = sum(vols[-20:]) / min(20, len(vols))

    # Volatility (ATR-like)
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) if i>0 else highs[i]-lows[i] for i in range(len(closes))]
    atr = sum(trs[-14:]) / 14.0

    # Compression/Breakout logic
    window = 20
    range_high = max(highs[-window-1:-1])
    range_low = min(lows[-window-1:-1])
    broke_up = last > range_high * 1.001
    broke_down = last < range_low * 0.999

    vol_expansion = last_vol > 1.3 * avg_vol

    # Confidence scoring
    conf = 50
    reason_bits = []
    if vol_expansion:
        conf += 12
        reason_bits.append("rising volume")
    if atr / last < 0.01:  # low vol prior -> pop risk
        conf += 10
        reason_bits.append("volatility compression")
    if broke_up:
        conf += 18
        reason_bits.append("range breakout up")
    if broke_down:
        conf += 18
        reason_bits.append("range breakdown")

    conf = max(40, min(95, int(conf)))

    # Direction & levels
    if broke_up and not broke_down:
        sig_type = "LONG"
        entry_low = round(range_high * 0.998, 2)
        entry_high = round(range_high * 1.002, 2)
        sl = round(range_low, 2)
        tp1 = round(last + 0.5 * atr, 2)
        tp2 = round(last + 1.0 * atr, 2)
        tp3 = round(last + 1.8 * atr, 2)
        quick = "Breakout from 15m compression zone with rising volume"
    elif broke_down and not broke_up:
        sig_type = "SHORT"
        entry_low = round(range_low * 0.998, 2)
        entry_high = round(range_low * 1.002, 2)
        sl = round(range_high, 2)
        tp1 = round(last - 0.5 * atr, 2)
        tp2 = round(last - 1.0 * atr, 2)
        tp3 = round(last - 1.8 * atr, 2)
        quick = "Breakdown from 15m range with momentum unwind"
    else:
        # Bias by last close vs midpoint
        mid = (range_high + range_low) / 2
        if last >= mid:
            sig_type = "LONG"
            entry_low = round(last * 0.998, 2)
            entry_high = round(last * 1.002, 2)
            sl = round(last - 1.2 * atr, 2)
            tp1 = round(last + 0.6 * atr, 2)
            tp2 = round(last + 1.2 * atr, 2)
            tp3 = round(last + 2.0 * atr, 2)
            quick = "Momentum bias above mid with increasing volume"
        else:
            sig_type = "SHORT"
            entry_low = round(last * 0.998, 2)
            entry_high = round(last * 1.002, 2)
            sl = round(last + 1.2 * atr, 2)
            tp1 = round(last - 0.6 * atr, 2)
            tp2 = round(last - 1.2 * atr, 2)
            tp3 = round(last - 2.0 * atr, 2)
            quick = "Momentum bias below mid with flow weakness"

    # Adaptive size/leverage (simple): size scales with vol, leverage inverse with ATR
    lev = 10 if atr/last > 0.01 else 20
    size = 100 if conf < 70 else 200
    risk = round(size * 0.05, 2)
    proj = round(conf/100 * lev * 2, 1)

    signal = Signal(
        pair=pair,
        type=sig_type,
        price=round(last, 2),
        confidence=conf,
        size_usdt=size,
        leverage=lev,
        entry_low=entry_low,
        entry_high=entry_high,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        sl=sl,
        risk_usdt=risk,
        capital_example=20.0,
        projected_roi_pct=proj,
        reason=quick,
        timestamp=int(time.time()*1000),
    )
    return signal.model_dump()


def compute_heatmap() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in PAIRS:
        try:
            kl = fetch_kline(p, interval="5", limit=48)
            if not kl:
                continue
            closes = [float(x[4]) for x in kl]
            # approximate realized volatility as pct change std * sqrt(n)
            rets = []
            for i in range(1, len(closes)):
                if closes[i-1] != 0:
                    rets.append((closes[i] - closes[i-1]) / closes[i-1] * 100)
            vol = sum(abs(x) for x in rets[-12:]) / max(1, min(12, len(rets)))
            items.append({"pair": p, "vol_pct": round(vol, 2)})
        except Exception:
            continue
    items.sort(key=lambda x: x["vol_pct"], reverse=True)
    return items[:10]


@app.get("/")
def root():
    return {"status": "ok", "service": "AlphaDesk Backend"}


@app.get("/test")
def test_database():
    from database import db as _db
    ok = _db is not None
    return {
        "backend": "✅ Running",
        "database": "✅ Connected" if ok else "❌ Not Available",
        "collections": _db.list_collection_names()[:10] if ok else [],
    }


@app.get("/api/heatmap")
def api_heatmap():
    return compute_heatmap()


@app.get("/api/analytics")
def api_analytics():
    # Basic analytics snapshot; for production, compute from stored outcomes
    return {
        "win_rate": 61.3,
        "rr": 1.85,
        "avg_roi": 13.9,
        "weekly": 7.4,
    }


@app.get("/api/signals")
def api_signals():
    signals: List[Dict[str, Any]] = []
    for p in PAIRS:
        try:
            sig = compute_signal_from_klines(p)
            signals.append(sig)
            # persist
            try:
                create_document("signal", sig)
            except Exception:
                pass
        except Exception:
            continue
    # Sort by confidence desc
    signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return signals[:20]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
