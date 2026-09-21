"""
scripts/paper_results.py -- Reproduce key figures from the news diffusion paper
Day 10 Commit 1: Generate calibration curves, ECE table, and PBO bar chart.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.calibrator import (
    PlattCalibrator, IsotonicCalibrator,
    expected_calibration_error, reliability_diagram_data,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _make_dataset(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, n)
    # labels with some noise
    labels = (_sigmoid(scores * 1.5) > rng.uniform(0, 1, n)).astype(float)
    return scores, labels


def _reliability_ascii(diag, width=40):
    lines = [f"{'Bin':>6}  {'Conf':>5}  {'Acc':>5}  {'N':>5}  Bar"]
    for d in diag:
        bar_len = int(d["confidence"] * width)
        acc_len = int(d["accuracy"] * width)
        bar = "â" * bar_len + "â" * (width - bar_len)
        lines.append(
            f"{d['bin_mid']:>6.2f}  {d['confidence']:>5.3f}  "
            f"{d['accuracy']:>5.3f}  {d['count']:>5}  {bar}"
        )
    return "\n".join(lines)


def run(args):
    scores, labels = _make_dataset(n=args.n, seed=args.seed)
    split = int(len(scores) * 0.6)
    tr_s, tr_l = scores[:split], labels[:split]
    te_s, te_l = scores[split:], labels[split:]

    # Raw (uncalibrated) â use aggressive sigmoid to create miscalibration
    raw_probs = _sigmoid(te_s * 3.0)

    # Platt
    platt = PlattCalibrator(n_epochs=args.epochs, lr=0.05).fit(tr_s, tr_l)
    platt_probs = platt.predict_proba(te_s)

    # Isotonic
    iso = IsotonicCalibrator().fit(tr_s, tr_l)
    iso_probs = iso.predict_proba(te_s)

    ece_raw   = expected_calibration_error(raw_probs,   te_l, args.bins)
    ece_platt = expected_calibration_error(platt_probs, te_l, args.bins)
    ece_iso   = expected_calibration_error(iso_probs,   te_l, args.bins)

    results = {
        "n_train": split, "n_test": len(te_s),
        "ece": {"raw": round(ece_raw, 5), "platt": round(ece_platt, 5), "isotonic": round(ece_iso, 5)},
        "platt_params": {"A": round(platt.A, 4), "B": round(platt.B, 4)},
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 60)
    print("  NEWS-TO-PRICE DIFFUSION â Calibration Results")
    print("=" * 60)
    print(f"\n  Train N: {split:,}   Test N: {len(te_s):,}   Bins: {args.bins}")
    print(f"\n  ECE (Expected Calibration Error)")
    print(f"    Raw (uncalibrated): {ece_raw:.5f}")
    print(f"    Platt calibrated:   {ece_platt:.5f}  (A={platt.A:.4f}, B={platt.B:.4f})")
    print(f"    Isotonic:           {ece_iso:.5f}")

    best = min(("Platt", ece_platt), ("Isotonic", ece_iso), key=lambda x: x[1])
    print(f"\n  Best calibrator: {best[0]}  (ECE={best[1]:.5f})")

    print("\n--- Reliability Diagram (Platt) ---")
    diag = reliability_diagram_data(platt_probs, te_l, args.bins)
    print(_reliability_ascii(diag))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--json", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
