"""
api/app.py -- FastAPI service for news-to-price diffusion calibration
news-to-price-diffusion Day 11
Endpoints: POST /calibrate, GET /health, POST /score
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import math
import json

# ââ Models ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

if HAS_FASTAPI:
    class CalibrateRequest(BaseModel):
        scores: List[float] = Field(..., description="Raw model scores (any range)")
        labels: List[float] = Field(..., description="Binary labels 0/1")
        method: str = Field("platt", description="'platt' or 'isotonic'")
        n_epochs: int = Field(1000, description="Epochs for Platt fitting")
        lr: float = Field(0.01, description="Learning rate for Platt fitting")
        n_bins: int = Field(10, description="Bins for ECE computation")

    class CalibrateResponse(BaseModel):
        method: str
        ece_before: float
        ece_after: float
        ece_improvement_pct: float
        platt_A: Optional[float] = None
        platt_B: Optional[float] = None
        n_isotonic_knots: Optional[int] = None
        elapsed_ms: float

    class ScoreRequest(BaseModel):
        scores: List[float]
        method: str = "platt"
        # Platt params (returned from /calibrate)
        platt_A: Optional[float] = None
        platt_B: Optional[float] = None
        # Isotonic calibration points
        iso_x: Optional[List[float]] = None
        iso_y: Optional[List[float]] = None

    class ScoreResponse(BaseModel):
        calibrated_probs: List[float]
        mean_prob: float
        elapsed_ms: float


# ââ Helpers (pure Python, no numpy required for basic case) ââââââââââââââââââ

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(x, 30.0))))


def _platt_predict(scores: List[float], A: float, B: float) -> List[float]:
    return [_sigmoid(A * s + B) for s in scores]


def _interp(x: float, xs: List[float], ys: List[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i+1]:
            t = (x - xs[i]) / (xs[i+1] - xs[i])
            return ys[i] + t * (ys[i+1] - ys[i])
    return ys[-1]


def _ece(probs: List[float], labels: List[float], n_bins: int = 10) -> float:
    n = len(probs)
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = [i for i, p in enumerate(probs) if lo <= p < hi]
        if lo == edges[-2]:
            mask = [i for i, p in enumerate(probs) if p >= lo]
        if not mask:
            continue
        conf = sum(probs[i] for i in mask) / len(mask)
        acc = sum(labels[i] for i in mask) / len(mask)
        ece += len(mask) / n * abs(conf - acc)
    return ece


# ââ App âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

if HAS_FASTAPI:
    app = FastAPI(
        title="News-to-Price Diffusion Calibration API",
        description="Platt and isotonic calibration for diffusion model probability outputs",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "calibration-api", "version": "1.0.0"}

    @app.post("/calibrate", response_model=CalibrateResponse)
    def calibrate(req: CalibrateRequest):
        t0 = time.monotonic()
        scores = req.scores
        labels = req.labels
        if len(scores) != len(labels):
            raise HTTPException(status_code=400, detail="scores and labels must be same length")
        if len(scores) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 samples")

        # Import calibrators
        try:
            from models.calibrator import (PlattCalibrator, IsotonicCalibrator,
                                           expected_calibration_error)
            import numpy as np
            scores_np = np.array(scores)
            labels_np = np.array(labels)
            raw_probs = np.array([_sigmoid(s) for s in scores])
            ece_before = float(expected_calibration_error(raw_probs, labels_np, req.n_bins))
            platt_A = platt_B = None
            n_knots = None

            if req.method == "platt":
                cal = PlattCalibrator(lr=req.lr, n_epochs=req.n_epochs).fit(scores_np, labels_np)
                cal_probs = cal.predict_proba(scores_np)
                platt_A, platt_B = float(cal.A), float(cal.B)
            else:
                cal = IsotonicCalibrator().fit(scores_np, labels_np)
                cal_probs = cal.predict_proba(scores_np)
                n_knots = len(cal._x) if cal._x is not None else 0

            ece_after = float(expected_calibration_error(cal_probs, labels_np, req.n_bins))
        except ImportError:
            # Fallback: simplified Platt via gradient descent
            A, B = 1.0, 0.0
            lr = req.lr
            for _ in range(req.n_epochs):
                probs = [_sigmoid(A * s + B) for s in scores]
                dA = sum((probs[i] - labels[i]) * scores[i] for i in range(len(scores))) / len(scores)
                dB = sum(probs[i] - labels[i] for i in range(len(scores))) / len(scores)
                A -= lr * dA
                B -= lr * dB
            raw_probs_list = [_sigmoid(s) for s in scores]
            cal_probs_list = [_sigmoid(A * s + B) for s in scores]
            ece_before = _ece(raw_probs_list, labels, req.n_bins)
            ece_after = _ece(cal_probs_list, labels, req.n_bins)
            platt_A, platt_B = A, B
            n_knots = None

        elapsed_ms = (time.monotonic() - t0) * 1000
        improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0

        return CalibrateResponse(
            method=req.method,
            ece_before=round(ece_before, 6),
            ece_after=round(ece_after, 6),
            ece_improvement_pct=round(improvement, 2),
            platt_A=round(platt_A, 6) if platt_A is not None else None,
            platt_B=round(platt_B, 6) if platt_B is not None else None,
            n_isotonic_knots=n_knots,
            elapsed_ms=round(elapsed_ms, 2),
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest):
        t0 = time.monotonic()
        if req.method == "platt":
            if req.platt_A is None or req.platt_B is None:
                raise HTTPException(400, "platt_A and platt_B required for Platt scoring")
            probs = _platt_predict(req.scores, req.platt_A, req.platt_B)
        elif req.method == "isotonic":
            if not req.iso_x or not req.iso_y:
                raise HTTPException(400, "iso_x and iso_y required for isotonic scoring")
            probs = [_interp(s, req.iso_x, req.iso_y) for s in req.scores]
        else:
            probs = [_sigmoid(s) for s in req.scores]
        mean_prob = sum(probs) / len(probs) if probs else 0.0
        elapsed_ms = (time.monotonic() - t0) * 1000
        return ScoreResponse(
            calibrated_probs=probs,
            mean_prob=round(mean_prob, 6),
            elapsed_ms=round(elapsed_ms, 2),
        )

else:
    # Stub for environments without FastAPI
    def create_app():
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")


if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Install fastapi and uvicorn: pip install fastapi uvicorn")
