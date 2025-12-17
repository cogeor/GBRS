# Base Score (`y0`) Computation in GBRS

This memo documents how the base score is computed for each objective type.

---

## Score Formula

For all objectives:
```
score = y0 + Σ(w[i] × 1{X[feature[i]] > threshold[i]})
```

Where:
- `y0` = base score (intercept)
- `w[i]` = weight for stump `i` (always ≥ 0)
- Indicator is 1 if feature exceeds threshold, 0 otherwise

**Minimum possible score** = `y0` (all features below thresholds)

---

## By Objective Type

### Continuous (Regression)

```cpp
y0 = 0  // Initial value
// During training: y0 += sum of all w2 values
```

**Interpretation**: Predicted value when all features are in "low" categories.

---

### Binary (Classification)

```cpp
y0 = logodds(y) = log(p / (1-p))  // where p = mean(y)
```

**Interpretation**: Log-odds of positive class for baseline patient.

**In bindings.cpp:**
```cpp
void fit_proba(...) {
    double y0 = logodds(y);  // Prior probability
    model->params.y0 = y0;
    model->fit_proba(...);
}
```

---

### Survival

```cpp
// After training, compute baseline for low-risk group:
VectorXd scores = predict(X);
double q25 = percentile(scores, 0.25);  // Bottom 25%

double events = 0, time = 0;
for (i = 0; i < n; i++) {
    if (scores[i] <= q25) {
        events += E[i];
        time += T[i];
    }
}
y0 = log(events / time);  // Log hazard rate
```

**Interpretation**: `y0` = log(event rate) for lowest-risk patients.

- A patient with score = `y0` has baseline hazard rate
- Each +1 point increases log-hazard by 1 unit
- Hazard ratio = `exp(score - y0)`

---

## Key Insight: Weight Structure

Weights are structured so that:
```
y0 += w2  (accumulated during training)
w[i] = w2 - w1  (stored weight)
```

This means:
- Weights are **always positive** (points add risk, never subtract)
- `y0` represents the **minimum score** (lowest-risk patient)
- **Score 0 ≠ median risk** — it's the *lowest possible risk*

---

## Summary Table

| Objective | y0 Computation | Interpretation |
|-----------|---------------|----------------|
| Continuous | `0 + Σ(w2)` | Mean prediction for baseline |
| Binary | `logodds(mean(y))` | Prior log-odds |
| Survival | `log(events_low / time_low)` | Log hazard rate for low-risk group |
