import numpy as np
from gbrs.core import Model

class GBRS:
    def __init__(self, n_iter=300, lr=0.05, n_quantiles=5, ss_rate=1.0):
        self._model = Model(n_iter, lr, n_quantiles, ss_rate)

    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def fit_proba(self, X, y):
        return self._model.fit_proba(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
    
    def fit_survival(self, X, time, event):
        """Fit the model for survival analysis."""
        return self._model.fit_survival(X, time, event)
    
    def print(self, feature_names=None):
        print_model(self._model, feature_names)

def prune_weights(idx_array, split_val_array, w_array):
    merged = {}

    for idx, split_val, w in zip(idx_array, split_val_array, w_array):
        key = (idx, split_val)
        if key in merged:
            merged[key] += w
        else:
            merged[key] = w

    if not merged:
        return (
            np.array([], dtype=idx_array.dtype),
            np.array([], dtype=split_val_array.dtype),
            np.array([], dtype=w_array.dtype),
        )

    keys = list(merged.keys())
    idx_out = np.array([k[0] for k in keys], dtype=idx_array.dtype)
    split_val_out = np.array([k[1] for k in keys], dtype=split_val_array.dtype)
    w_out = np.array([merged[k] for k in keys], dtype=w_array.dtype)

    return idx_out, split_val_out, w_out

def get_score_breaks(split_val, idx_array, w_array, idx, prec=1):
    # Filter rows where idx == (idx - 1) like in R
    mask = (idx_array == idx)
    vals_split = split_val[mask]
    vals_w = w_array[mask]

    if vals_split.size == 0:
        return {}

    # Single value case
    if vals_split.size == 1:
        return {
            "index": idx,
            "breaks": [f"<{vals_split[0]:.{prec}f}"],
            "weights": [f"{vals_w[0]:.{prec}f}"]
        }

    # Sort by split_val
    sorted_indices = np.argsort(vals_split)
    vals_split = vals_split[sorted_indices]
    vals_w = vals_w[sorted_indices]

    n = vals_split.shape[0]
    weights = np.zeros(n + 1)

    for i in range(n):
        w = vals_w[i]
        if w > 0:
            weights[i+1:] += w
        else:
            weights[:i+1] -= w

    weights_fmt = [f"{w:.{prec}f}" for w in weights]
    splits_fmt = [f"{v:.{prec}f}" for v in vals_split]

    # Binary case
    if n == 1 and vals_split[0] == 0:
        return {
            "index": idx,
            "weights": weights_fmt,
            "breaks": ["FALSE", "TRUE"]
        }

    # General case
    breaks = [f"<{splits_fmt[0]}"]
    for i in range(1, len(splits_fmt)):
        breaks.append(f"[{splits_fmt[i-1]},{splits_fmt[i]})")
    breaks.append(f">={splits_fmt[-1]}")

    return {
        "index": idx,
        "weights": weights_fmt,
        "breaks": breaks
    }

def print_score_table(score_breaks_dict):
    for idx, result in score_breaks_dict.items():
        if not result or not result.get("breaks"):
            continue
        
        name = result.get("feature_name") or result.get("name") or f"F {int(idx)}"

        breaks = result["breaks"]
        weights = result["weights"]
        
        col_widths = [max(len(str(b)), len(str(w))) for b, w in zip(breaks, weights)]
        name_col_width = max(len(name), 7)
        total_width = name_col_width + 3 + sum(col_widths) + 3 * len(col_widths) + 1

        print("=" * total_width)
        
        header = f"| {name}".ljust(name_col_width + 2)
        for i, b in enumerate(breaks):
            header += f"| {b}".ljust(col_widths[i] + 3)
        print(header + "|")

        weights_row = " " * (name_col_width + 3)
        for i, w in enumerate(weights):
            weights_row += f"| {w}".ljust(col_widths[i] + 3)
        print(weights_row + "|")

        print("=" * total_width)
def build_score_breaks_dict(split_val, idx, w, indices, feature_names=None):
    """
    Build a dict mapping each index to its score breaks dict.
    
    Parameters:
        split_val (array-like): Array of split values.
        idx (array-like): Array of indices corresponding to features.
        w (array-like): Array of weights.
        indices (iterable): All feature indices to process.
        feature_names (dict, optional): Mapping from index to feature name.
    
    Returns:
        dict: {index: {"index": index, "weights": [...], "breaks": [...], "feature_name": str or None}}
    """
    result = {}
    for i in indices:
        score_breaks = get_score_breaks(split_val, idx, w, i)
        if score_breaks and score_breaks.get("breaks"):  # Only add non-empty results
            if feature_names:
                score_breaks["feature_name"] = feature_names.get(i, None)
            else:
                score_breaks["feature_name"] = None
            result[i] = score_breaks
    return result

def print_model(model, feature_names=None):
    """
    Print the model score table.
    
    Parameters:
        model: The model object (C++ wrapper).
        feature_names (dict, optional): Mapping from index to feature name.
    """
    params = model.get_params()
    idxs = model.get_idxs()
    split_vals = model.get_split_val()
    
    # Prune weights (aggregate weights with same idx and split_val)
    idxs, split_vals, w = prune_weights(idxs, split_vals, params.w)
    
    if feature_names:
        indices = sorted(feature_names.keys())
    else:
        indices = np.unique(idxs)
        indices.sort()
        
    d = build_score_breaks_dict(split_vals, idxs, w, indices, feature_names)
    print_score_table(d)