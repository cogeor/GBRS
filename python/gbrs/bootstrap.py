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

    def _collect_weight_matrix(
        self,
    ) -> Tuple[Dict[int, List[float]], Dict[int, NDArray[np.float64]]]:
        """Build per-feature weight matrices from bootstrap samples.

        Returns (by_feature_thresholds, by_feature_weights) where:
        - by_feature_thresholds maps feature_idx -> sorted list of threshold values
        - by_feature_weights maps feature_idx -> 2D array of shape (n_bootstrap, n_thresholds)
        """
        all_keys: set = set()
        for weights in self.all_weights:
            all_keys.update(weights.keys())

        by_feature: Dict[int, set] = {}
        for idx, sv in all_keys:
            if idx not in by_feature:
                by_feature[idx] = set()
            by_feature[idx].add(sv)

        thresholds_map: Dict[int, List[float]] = {}
        weights_map: Dict[int, NDArray[np.float64]] = {}

        for idx in sorted(by_feature.keys()):
            thresholds = sorted(by_feature[idx])
            thresholds_map[idx] = thresholds
            sv_to_col = {sv: j for j, sv in enumerate(thresholds)}

            mat = np.zeros((self.n_bootstrap, len(thresholds)))
            for i, weights in enumerate(self.all_weights):
                for (widx, sv), w in weights.items():
                    if widx == idx:
                        mat[i, sv_to_col[sv]] = w

            weights_map[idx] = mat

        return thresholds_map, weights_map

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
        thresholds_map, weights_map = self._collect_weight_matrix()
        result: Dict[int, Dict[str, Any]] = {}
        for idx in sorted(thresholds_map.keys()):
            mat = weights_map[idx]
            name = f"F{idx}"
            if feature_names and idx in feature_names:
                name = feature_names[idx]
            result[idx] = {
                "thresholds": thresholds_map[idx],
                "mean": np.mean(mat, axis=0).tolist(),
                "std": np.std(mat, axis=0, ddof=1).tolist(),
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
        thresholds_map, weights_map = self._collect_weight_matrix()
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        result: Dict[int, Dict[str, Any]] = {}
        for idx in sorted(thresholds_map.keys()):
            mat = weights_map[idx]
            name = f"F{idx}"
            if feature_names and idx in feature_names:
                name = feature_names[idx]
            result[idx] = {
                "thresholds": thresholds_map[idx],
                "lower": np.percentile(mat, lower_pct, axis=0).tolist(),
                "upper": np.percentile(mat, upper_pct, axis=0).tolist(),
                "median": np.median(mat, axis=0).tolist(),
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
