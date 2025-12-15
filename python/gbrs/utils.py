from typing import Optional, Dict, Any, Tuple, List
from numpy.typing import NDArray
import numpy as np
from gbrs.core import Model

class GBRS:
    def __init__(self, n_iter: int = 300, lr: float = 0.05, n_quantiles: int = 5, batch_size: int = 0) -> None:
        self._model = Model(n_iter, lr, n_quantiles, batch_size)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], user_quantiles: Optional[list] = None) -> None:
        return self._model.fit(X, y, user_quantiles)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.predict(X)

    def fit_proba(self, X: NDArray[np.float64], y: NDArray[np.float64], user_quantiles: Optional[list] = None) -> None:
        return self._model.fit_proba(X, y, user_quantiles)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._model.predict_proba(X)
    
    def fit_survival(self, X: NDArray[np.float64], time: NDArray[np.float64], event: NDArray[np.float64], user_quantiles: Optional[list] = None) -> None:
        """Fit the model for survival analysis."""
        return self._model.fit_survival(X, time, event, user_quantiles)
    
    def print(self, feature_names: Optional[Dict[int, str]] = None, format: str = "ascii_h") -> None:
        """
        Print the model score table.
        
        Parameters
        ----------
        feature_names : dict, optional
            Mapping from index to feature name.
        format : str, optional
            Output format: "text" (default), "latex", "md", "latex_h", "md_h", "ascii_h".
        """
        print_model(self._model, feature_names, format)

    def print_vertical(self, feature_names: Optional[Dict[int, str]] = None) -> None:
        """
        Print the model score table in the legacy vertical format.
        
        Parameters
        ----------
        feature_names : dict, optional
            Mapping from index to feature name.
        """
        self.print(feature_names, format="text")

    def save_model(self, filepath: str) -> None:
        """
        Save the model to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model file
        """
        from gbrs.model_io import save_model as _save_model
        _save_model(self, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "GBRS":
        """
        Load a model from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the model file
            
        Returns
        -------
        GBRS
            Loaded model instance
        """
        from gbrs.model_io import load_model as _load_model
        return _load_model(filepath)

    def _set_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Set model state from a dictionary.
        
        Parameters
        ----------
        state_dict : dict
            Dictionary containing model parameters
        """
        rules = state_dict["rules"]
        
        idxs = []
        split_vals = []
        w = []
        y0 = 0.0
        
        for i, rule in enumerate(rules):
            idxs.append(rule["idx"])
            split_vals.append(rule["split_val"])
            w.append(rule["w"])
            if i == 0:
                y0 = rule.get("cst", 0.0)
        
        import numpy as np
        idxs_arr = np.array(idxs, dtype=np.float64)
        split_vals_arr = np.array(split_vals, dtype=np.float64)
        w_arr = np.array(w, dtype=np.float64)
        
        self._model.set_params(idxs_arr, split_vals_arr, w_arr, float(y0))

def prune_weights(idx_array: NDArray[np.float64], split_val_array: NDArray[np.float64], w_array: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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

def get_score_breaks(split_val: NDArray[np.float64], idx_array: NDArray[np.float64], w_array: NDArray[np.float64], idx: float, prec: int = 1) -> Dict[str, Any]:
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

def print_score_table(score_breaks_dict: Dict[float, Dict[str, Any]], base_score: Optional[float] = None) -> None:
    if base_score is not None:
        print(f"Base Score: {base_score:.4f}")
    
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

def print_latex_horizontal(score_breaks_dict: Dict[float, Dict[str, Any]]) -> None:
    # Calculate max columns needed
    max_cols = 0
    for idx, result in score_breaks_dict.items():
        if result and result.get("breaks"):
            max_cols = max(max_cols, len(result["breaks"]))

    col_spec = "l" + "l" * max_cols
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline")

    for idx, result in score_breaks_dict.items():
        if not result or not result.get("breaks"):
            continue
            
        name = result.get("feature_name") or result.get("name") or f"F {int(idx)}"
        breaks = result["breaks"]
        weights = result["weights"]
        
        # Escape LaTeX
        breaks = [b.replace("<", "\\ensuremath{<} ").replace(">=", "\\ensuremath{\\ge} ").replace(">", "\\ensuremath{>} ") for b in breaks]
        
        # Pad
        n_pad = max_cols - len(breaks)
        breaks_padded = breaks + [""] * n_pad
        weights_padded = weights + [""] * n_pad
        
        # Row 1
        row1 = f"{name} & " + " & ".join(breaks_padded) + " \\\\"
        print(row1)
        
        # Row 2
        row2 = " & " + " & ".join(weights_padded) + " \\\\"
        print(row2)
        
    print("\\hline")
    print("\\end{tabular}")

def print_md_horizontal(score_breaks_dict: Dict[float, Dict[str, Any]]) -> None:
    # Calculate max columns needed
    max_cols = 0
    for idx, result in score_breaks_dict.items():
        if result and result.get("breaks"):
            max_cols = max(max_cols, len(result["breaks"]))

    header = "| Variable | " + " | " * max_cols + "|"
    separator = "|:---|" + ":---|" * max_cols + "|"
    print(header)
    print(separator)

    for idx, result in score_breaks_dict.items():
        if not result or not result.get("breaks"):
            continue
            
        name = result.get("feature_name") or result.get("name") or f"F {int(idx)}"
        breaks = result["breaks"]
        weights = result["weights"]
        
        # Pad
        n_pad = max_cols - len(breaks)
        breaks_padded = breaks + [""] * n_pad
        weights_padded = weights + [""] * n_pad
        
        # Row 1
        row1 = f"| **{name}** | " + " | ".join(breaks_padded) + " |"
        print(row1)
        
        # Row 2
        row2 = "| | " + " | ".join(weights_padded) + " |"
        print(row2)


def print_ascii_horizontal(score_breaks_dict: Dict[float, Dict[str, Any]], base_score: Optional[float] = None) -> None:
    if base_score is not None:
        print(f"Base Score: {base_score:.4f}")
        print("-" * 20)
        print()

    rows = []
    for idx, result in score_breaks_dict.items():
        if result and result.get("breaks"):
            rows.append({
                "name": result.get("feature_name") or result.get("name") or f"F {int(idx)}",
                "breaks": result["breaks"],
                "weights": result["weights"]
            })
            
    if not rows:
        return
        
    max_bins = max(len(r["breaks"]) for r in rows)
    col_widths = [0] * (max_bins + 1)
    
    # Variable name width
    col_widths[0] = max(len(r["name"]) for r in rows)
    
    # Bin widths
    for j in range(max_bins):
        w = 0
        for r in rows:
            if len(r["breaks"]) > j:
                w = max(w, len(r["breaks"][j]), len(r["weights"][j]))
        col_widths[j+1] = w
        
    # Padding within the cell
    col_widths = [w + 2 for w in col_widths]
    
    def print_row(cols, is_header=False):
        line = ""
        for j, val in enumerate(cols):
            width = col_widths[j]
            # Content padded
            cell_content = val.ljust(width)
            if j == 0:
                line += cell_content
            else:
                # Add separator before the column (except first one, which is Name)
                # But wait, we want separators between cutoffs.
                # Cutoffs start at index 1.
                 line += "| " + cell_content
        print(line)

    for i, r in enumerate(rows):
        # Row 1 (Header/Breaks)
        cols1 = [r["name"]] + r["breaks"]
        if len(cols1) < len(col_widths):
            cols1 += [""] * (len(col_widths) - len(cols1))
        print_row(cols1, is_header=True)
        
        # Row 2 (Weights)
        cols2 = [""] + r["weights"]
        if len(cols2) < len(col_widths):
            cols2 += [""] * (len(col_widths) - len(cols2))
        print_row(cols2)
        
        # Separation between features
        # Add a light separation (e.g. dashed line) if not the last one
        if i < len(rows) - 1:
             # Calculate total length roughly
             # Sum of widths + number of separators * 2
             total_len = sum(col_widths) + (len(col_widths) - 1) * 2
             print("-" * total_len)
        else:
             print() # formatting

def build_score_breaks_dict(split_val: NDArray[np.float64], idx: NDArray[np.float64], w: NDArray[np.float64], indices: NDArray[np.float64], feature_names: Optional[Dict[int, str]] = None) -> Dict[float, Dict[str, Any]]:
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

def print_model(model: Model, feature_names: Optional[Dict[int, str]] = None, format: str = "ascii_h") -> None:
    """
    Print the model score table.
    
    Parameters:
        model: The model object (C++ wrapper).
        feature_names (dict, optional): Mapping from index to feature name.
        format: Output format ("text", "latex", "md", "latex_h", "md_h", "ascii_h")
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
    
    if format == "text":
        print_score_table(d, base_score=params.y0)
    elif format == "latex_h":
        print_latex_horizontal(d)
    elif format == "md_h":
        print_md_horizontal(d)
    elif format == "ascii_h":
        print_ascii_horizontal(d, base_score=params.y0)
    else:
        # Fallback to text for now if other formats not implemented or requested
        print_score_table(d, base_score=params.y0)