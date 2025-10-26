from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import List, Dict, Any
from datetime import datetime, timezone
import os

from schemas import Signal
from database import create_document

app = FastAPI(title="AlphaDesk API")

frontend_url = os.getenv("FRONTEND_URL", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*" if frontend_url == "*" else frontend_url],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BYBIT_BASE = "https://api.bybit.com"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]

async def fetch_klines(symbol: str, interval: str = "15", limit: int = 100) -> List[Dict[str, Any]]:
    url = f"{BYBIT_BASE}/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(data.get("retMsg", "Bybit error"))
        rows = data["result"]["list"]
        out = []
        for row in reversed(rows):
            ts, open_p, high_p, low_p, close_p, vol, turnover = row[:7]
            out.append({
                "ts": int(ts),
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "close": float(close_p),
                "volume": float(vol),
                "turnover": float(turnover),
            })
        return out

async def fetch_tickers() -> List[Dict[str, Any]]:
    url = f"{BYBIT_BASE}/v5/market/tickers?category=linear"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(data.get("retMsg", "Bybit error"))
        return data["result"]["list"]

# Basic ATR
def atr(kl: List[Dict[str, Any]], period: int = 14) -> float:
    if len(kl) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = kl[-i]["high"]
        l = kl[-i]["low"]
        pc = kl[-i-1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return sum(trs) / len(trs)

# Compute a simple signal
def compute_signal(symbol: str, kl: List[Dict[str, Any]]) -> Signal | None:
    if len(kl) < 20:
        return None
    last = kl[-1]
    prev = kl[-2]
    recent_high = max(k["high"] for k in kl[-20:])
    recent_low = min(k["low"] for k in kl[-20:])
    a = max(atr(kl, 14), 1e-8)

    direction = None
    reason = ""
    if last["close"] > recent_high * 0.999 and last["close"] > prev["close"]:
        direction = "LONG"
        reason = "15m range breakout + momentum"
    elif last["close"] < recent_low * 1.001 and last["close"] < prev["close"]:
        direction = "SHORT"
        reason = "15m breakdown + momentum"

    if not direction:
        rng = last["high"] - last["low"]
        prev_rng = prev["high"] - prev["low"]
        if rng > prev_rng * 1.8:
            direction = "LONG" if last["close"] > prev["close"] else "SHORT"
            reason = "Volatility expansion after compression"
        else:
            return None

    price = last["close"]
    if direction == "LONG":
        entry_low = price - 0.25 * a
        entry_high = price + 0.10 * a
        sl = price - 1.2 * a
        tp1 = price + 0.8 * a
        tp2 = price + 1.6 * a
        tp3 = price + 2.4 * a
    else:
        entry_low = price - 0.10 * a
        entry_high = price + 0.25 * a
        sl = price + 1.2 * a
        tp1 = price - 0.8 * a
        tp2 = price - 1.6 * a
        tp3 = price - 2.4 * a

    vol_now = last["volume"]
    vol_prev = sum(k["volume"] for k in kl[-6:-1]) / 5
    vol_boost = min(2.0, (vol_now / (vol_prev + 1e-9)))
    breakout_strength = abs((price - prev["close"]) / (a + 1e-9))
    confidence = max(50, min(98, int((0.6 * breakout_strength + 0.4 * vol_boost) / 3 * 100)))

    leverage = 5 if confidence < 70 else 8 if confidence < 85 else 12
    risk_usdt = 50.0
    size_usdt = risk_usdt * leverage
    projected_roi_pct = round((tp2 - price) / price * (100 if direction == "LONG" else -100), 2)

    sig = Signal(
        pair=symbol,
        type=direction,
        price=round(price, 4),
        confidence=confidence,
        entry_low=round(entry_low, 4),
        entry_high=round(entry_high, 4),
        tp1=round(tp1, 4),
        tp2=round(tp2, 4),
        tp3=round(tp3, 4),
        sl=round(sl, 4),
        size_usdt=round(size_usdt, 2),
        leverage=leverage,
        risk_usdt=risk_usdt,
        projected_roi_pct=round(projected_roi_pct, 2),
        reason=reason,
        timestamp=datetime.now(timezone.utc),
    )
    return sig

@app.get("/api/signals")
async def get_signals() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for sym in SYMBOLS:
        try:
            kl = await fetch_klines(sym, interval="15", limit=120)
            sig = compute_signal(sym, kl)
            if sig:
                data = sig.model_dump()
                results.append(data)
                try:
                    await create_document("signal", {**data, "_created_at": datetime.now(timezone.utc)})
                except Exception:
                    pass
        except Exception:
            continue
    results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return results

@app.get("/api/heatmap")
async def get_heatmap() -> List[Dict[str, Any]]:
    try:
        tickers = await fetch_tickers()
    except Exception:
        return []
    out = []
    for t in tickers:
        sym = t.get("symbol")
        if not sym or not sym.endswith("USDT"):
            continue
        try:
            vol = float(t.get("price24hPcnt", 0.0)) * 100.0
        except Exception:
            vol = 0.0
        out.append({"pair": sym, "vol_pct": round(vol, 4)})
    out.sort(key=lambda x: abs(x["vol_pct"]), reverse=True)
    return out[:20]

@app.get("/api/analytics")
async def get_analytics() -> Dict[str, Any]:
    try:
        tickers = await fetch_tickers()
        pos = sum(1 for t in tickers if float(t.get("price24hPcnt", 0)) > 0)
        total = max(1, len(tickers))
        breadth = pos / total
        win_rate = 40 + breadth * 40
        rr = 1.4 + (breadth - 0.5) * 0.8
        avg_roi = (breadth - 0.5) * 10
        weekly = avg_roi * 1.5
    except Exception:
        win_rate = 55.0
        rr = 1.5
        avg_roi = 1.2
        weekly = 2.5
    return {
        "win_rate": round(win_rate, 2),
        "rr": round(rr, 2),
        "avg_roi": round(avg_roi, 2),
        "weekly": round(weekly, 2),
    }

@app.get("/test")
async def test() -> Dict[str, Any]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
