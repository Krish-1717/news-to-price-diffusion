"""
run_demos.py — News-to-Price Diffusion
Runs all Day 19-30 demo modules in sequence.
Pure Python stdlib only — no external dependencies.
"""
import subprocess
import sys
import os
import time

MODULES = [
    "quant_code/news_day19_coverage_analysis.py",
    "quant_code/news_day20_consistency_model.py",
    "quant_code/news_day21_online_learning.py",
    "quant_code/news_day22_model_card.py",
    "quant_code/news_day23_latent_factor_diffusion.py",
    "quant_code/news_day24_causal_analysis.py",
    "quant_code/news_day25_sentiment_analysis.py",
    "quant_code/news_day26_multimodal_fusion.py",
    "quant_code/news_day27_event_study.py",
    "quant_code/news_day28_signal_backtest.py",
    "quant_code/news_day29_topic_modeling.py",
    "quant_code/news_day30_production_pipeline.py",
]

def run_module(path: str) -> bool:
    name = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗ {name} failed (exit code {result.returncode})")
        return False

if __name__ == "__main__":
    print("News-to-Price Diffusion — Day 19-30 Demo Runner")
    print("=" * 60)

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    passed, failed = 0, 0
    for module in MODULES:
        if run_module(module):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(MODULES)} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
