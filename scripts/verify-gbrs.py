"""End-to-end functional verification of gbrs.

Runs all three objectives, exercises bootstrap, round-trips a saved model,
and compares against a sklearn baseline where applicable. Exits non-zero on
any failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import gbrs
from gbrs import GBRS, load_model, save_model


RNG = np.random.default_rng(42)


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        raise SystemExit(f"verification failed: {msg}")


def gen_regression(n: int = 400, d: int = 6) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.uniform(0, 1, size=(n, d))
    # latent score depends on first 3 features non-linearly
    score = 2 * (X[:, 0] > 0.5) + 1.5 * (X[:, 1] > 0.7) - 1 * (X[:, 2] > 0.3)
    y = score + 0.1 * RNG.standard_normal(n)
    return X, y


def gen_binary(n: int = 400, d: int = 6) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.uniform(0, 1, size=(n, d))
    logit = 3 * X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (RNG.uniform(0, 1, size=n) < p).astype(float)
    return X, y


def gen_survival(n: int = 400, d: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Strong, monotone hazard signal: first feature dominates.
    X = RNG.uniform(0, 1, size=(n, d))
    log_hazard = 2.5 * X[:, 0] + 1.2 * X[:, 1] - 1.0 * X[:, 2]
    hazard = np.exp(log_hazard - log_hazard.mean())
    t_event = RNG.exponential(scale=1.0 / hazard)
    t_censor = RNG.exponential(scale=5.0, size=n)
    time = np.minimum(t_event, t_censor)
    status = (t_event <= t_censor).astype(float)
    return X, time, status


def test_regression() -> None:
    banner("Regression")
    X, y = gen_regression()
    model = GBRS(n_iter=200, lr=0.05, n_quantiles=4)
    model.fit(X, y)
    preds = model.predict(X)
    check(preds.shape == y.shape, f"prediction shape {preds.shape} matches y {y.shape}")
    mse = float(np.mean((preds - y) ** 2))
    # baseline: predicting the mean
    baseline_mse = float(np.var(y))
    print(f"  GBRS MSE: {mse:.4f}   mean-baseline MSE: {baseline_mse:.4f}")
    check(mse < 0.5 * baseline_mse, "GBRS beats mean baseline by >2x")


def test_binary() -> None:
    banner("Binary classification")
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    X, y = gen_binary()
    model = GBRS(n_iter=200, lr=0.05, n_quantiles=4)
    model.fit_proba(X, y)
    probs = model.predict_proba(X)
    check(probs.shape == y.shape, "predict_proba shape matches y")
    check(
        bool(np.all((probs >= 0) & (probs <= 1))),
        "all predicted probabilities in [0, 1]",
    )
    auc = float(roc_auc_score(y, probs))
    baseline_auc = float(
        roc_auc_score(
            y, LogisticRegression(max_iter=500).fit(X, y).predict_proba(X)[:, 1]
        )
    )
    print(f"  GBRS AUC: {auc:.4f}   LogReg AUC: {baseline_auc:.4f}")
    check(auc > 0.75, f"GBRS AUC ({auc:.3f}) > 0.75 on synthetic data")
    check(auc >= baseline_auc - 0.05, "GBRS within 0.05 AUC of LogReg baseline")


def test_survival() -> None:
    banner("Survival")
    X, time, status = gen_survival()
    model = GBRS(n_iter=200, lr=0.05, n_quantiles=4)
    model.fit_survival(X, time, status)
    risk = model.predict(X)
    check(risk.shape == time.shape, "risk shape matches time")

    # concordance: higher risk should correspond to shorter event time
    n_concordant, n_comparable = 0, 0
    for i in range(len(time)):
        if status[i] != 1:
            continue
        for j in range(len(time)):
            if time[j] > time[i]:
                n_comparable += 1
                if risk[i] > risk[j]:
                    n_concordant += 1
                elif risk[i] == risk[j]:
                    n_concordant += 0.5
    c_index = n_concordant / n_comparable if n_comparable else float("nan")
    print(f"  C-index: {c_index:.4f}   (random ~0.5, perfect = 1.0)")
    check(c_index > 0.6, f"C-index ({c_index:.3f}) > 0.6")


def test_bootstrap() -> None:
    banner("Bootstrap CIs")
    X, y = gen_binary(n=200)
    model = GBRS(n_iter=100, lr=0.05, n_quantiles=4)
    result = model.bootstrap_proba(X, y, n_bootstrap=20, random_state=7)
    cis = result.get_confidence_intervals(alpha=0.05)
    check(cis is not None, "bootstrap returned CIs")
    print(f"  bootstrap CI object: {type(cis).__name__}")


def test_save_load() -> None:
    banner("Save/load round-trip")
    X, y = gen_binary(n=200)
    model = GBRS(n_iter=100, lr=0.05, n_quantiles=4)
    model.fit_proba(X, y)
    probs_before = model.predict_proba(X)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.json"
        save_model(model, str(path))
        check(
            path.exists() and path.stat().st_size > 0,
            f"model file written ({path.stat().st_size} bytes)",
        )
        reloaded = load_model(str(path))
        probs_after = reloaded.predict_proba(X)
        max_diff = float(np.max(np.abs(probs_before - probs_after)))
        print(f"  max |before - after| = {max_diff:.2e}")
        check(max_diff < 1e-10, "predictions identical after reload")


def test_print() -> None:
    banner("Interpretable score table")
    X, y = gen_binary(n=200)
    model = GBRS(n_iter=50, lr=0.05, n_quantiles=4)
    model.fit_proba(X, y)
    feature_names = {i: f"feat_{i}" for i in range(X.shape[1])}
    print("  model.print() output:")
    model.print(feature_names)
    check(True, "print() did not raise")


def main() -> int:
    print(f"gbrs version: {gbrs.__version__}")
    test_regression()
    test_binary()
    test_survival()
    test_bootstrap()
    test_save_load()
    test_print()
    banner("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
