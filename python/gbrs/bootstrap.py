"""Bootstrapping support for GBRS models."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class BootstrapResult:
    """Results from bootstrapped GBRS fitting.

    Attributes
    ----------
    thresholds : list of arrays
        Fixed thresholds (quantiles) used for each feature across all bootstrap samples.
    all_weights : list of dicts
        Weights from each bootstrap iteration. Each dict maps (feature_idx, threshold) to weight.
    all_y0 : list of floats
        Base scores from each bootstrap iteration.
    objective : str
        The objective function used ('continuous', 'binary', 'survival').
    """

    thresholds: List[NDArray[np.float64]]
    all_weights: List[Dict[Tuple[int, float], float]]
    all_y0: List[float]
    objective: str = "continuous"

    @property
    def n_bootstrap(self) -> int:
        """Number of bootstrap samples."""
        return len(self.all_weights)

    def get_weight_stats(
        self, feature_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get mean ± std for weights at each threshold.

        Parameters
        ----------
        feature_names : dict, optional
            Mapping from feature index to feature name.

        Returns
        -------
        dict
            Dictionary mapping feature index to stats:
            {"thresholds": [...], "mean": [...], "std": [...], "name": str}
        """
        # Collect all unique (idx, split_val) keys across all bootstrap samples
        all_keys: set = set()
        for weights in self.all_weights:
            all_keys.update(weights.keys())

        # Group by feature index
        by_feature: Dict[int, Dict[float, List[float]]] = {}
        for idx, sv in all_keys:
            if idx not in by_feature:
                by_feature[idx] = {}
            by_feature[idx][sv] = []

        # Collect weights for each (idx, sv) across bootstrap samples
        # If a threshold wasn't selected in a bootstrap run, it gets weight 0
        for weights in self.all_weights:
            for idx, sv in all_keys:
                w = weights.get((idx, sv), 0.0)
                by_feature[idx][sv].append(w)

        # Compute statistics
        result: Dict[int, Dict[str, Any]] = {}
        for idx, sv_weights in sorted(by_feature.items()):
            thresholds = sorted(sv_weights.keys())
            weights_arrays = [np.array(sv_weights[sv]) for sv in thresholds]
            means = [float(np.mean(w)) for w in weights_arrays]
            stds = [float(np.std(w)) for w in weights_arrays]

            name = f"F{idx}"
            if feature_names and idx in feature_names:
                name = feature_names[idx]

            result[idx] = {
                "thresholds": thresholds,
                "mean": means,
                "std": stds,
                "name": name,
            }

        return result

    def get_y0_stats(self) -> Tuple[float, float]:
        """Get mean ± std for base score (y0).

        Returns
        -------
        tuple
            (mean, std) of base scores across bootstrap samples.
        """
        return float(np.mean(self.all_y0)), float(np.std(self.all_y0))

    def get_confidence_intervals(
        self, alpha: float = 0.05, feature_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Get percentile-based confidence intervals for weights.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Returns (alpha/2, 1-alpha/2) percentiles.
        feature_names : dict, optional
            Mapping from feature index to feature name.

        Returns
        -------
        dict
            Dictionary mapping feature index to CI stats:
            {"thresholds": [...], "lower": [...], "upper": [...], "median": [...]}
        """
        all_keys: set = set()
        for weights in self.all_weights:
            all_keys.update(weights.keys())

        by_feature: Dict[int, Dict[float, List[float]]] = {}
        for idx, sv in all_keys:
            if idx not in by_feature:
                by_feature[idx] = {}
            by_feature[idx][sv] = []

        for weights in self.all_weights:
            for idx, sv in all_keys:
                w = weights.get((idx, sv), 0.0)
                by_feature[idx][sv].append(w)

        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        result: Dict[int, Dict[str, Any]] = {}
        for idx, sv_weights in sorted(by_feature.items()):
            thresholds = sorted(sv_weights.keys())
            weights_arrays = [np.array(sv_weights[sv]) for sv in thresholds]

            lower = [float(np.percentile(w, lower_pct)) for w in weights_arrays]
            upper = [float(np.percentile(w, upper_pct)) for w in weights_arrays]
            median = [float(np.median(w)) for w in weights_arrays]

            name = f"F{idx}"
            if feature_names and idx in feature_names:
                name = feature_names[idx]

            result[idx] = {
                "thresholds": thresholds,
                "lower": lower,
                "upper": upper,
                "median": median,
                "name": name,
            }

        return result

    def print_summary(
        self, feature_names: Optional[Dict[int, str]] = None, prec: int = 3
    ) -> None:
        """
        Print formatted bootstrap results showing mean ± std for each weight.

        Parameters
        ----------
        feature_names : dict, optional
            Mapping from feature index to feature name.
        prec : int, default=3
            Number of decimal places to display.
        """
        print(f"GBRS Bootstrap Results ({self.n_bootstrap} samples)")
        print("=" * 50)

        y0_mean, y0_std = self.get_y0_stats()
        print(f"\nBase Score: {y0_mean:.{prec}f} ± {y0_std:.{prec}f}")
        print("-" * 50)

        stats = self.get_weight_stats(feature_names)

        for idx in sorted(stats.keys()):
            info = stats[idx]
            print(f"\n{info['name']}:")
            for i, sv in enumerate(info["thresholds"]):
                m, s = info["mean"][i], info["std"][i]
                # Format: "> threshold: +mean ± std"
                sign = "+" if m >= 0 else ""
                print(f"  > {sv:.{prec}f}: {sign}{m:.{prec}f} ± {s:.{prec}f}")

        print()
