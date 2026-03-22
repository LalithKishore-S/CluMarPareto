import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import spearmanr
from numpy.polynomial import polynomial as P
from sklearn.datasets import make_classification


class XDBSCAN_Clustering():
    def __init__(self):
        self.D            = None
        self.n_features   = None
        self.eps          = 0
        self.min_samples  = 0
        self.n_clusters   = 0
        self.noise_pct    = 0
        self.clusters     = {}
        self.noise        = []
        self.optimal_k    = None      # the K chosen by X-DBSCAN
        self._eps_list    = []        # Eps_K for each K
        self._minpts_list = []        # MinPts_K for each K

    
    def build_distance_matrix(self, data, use_spearman=False):
        if use_spearman:
            corr_matrix, _ = spearmanr(data, axis=0)
            corr_matrix = np.array(corr_matrix)
            if corr_matrix.ndim == 0:
                corr_matrix = np.array([[1.0, float(corr_matrix)],
                                        [float(corr_matrix), 1.0]])
        else:
            corr_matrix = np.corrcoef(data.T)

        corr_matrix  = np.clip(corr_matrix, -1.0, 1.0)
        self.D       = np.sqrt(1.0 - np.square(corr_matrix))
        np.fill_diagonal(self.D, 0.0)
        self.n_features = self.D.shape[0]

    
    def _kdist_curve(self, k):
        n = self.n_features
        k = min(k, n - 1)
        sorted_D   = np.sort(self.D, axis=1)
        kdist_vals = sorted_D[:, k]           
        return np.sort(kdist_vals)            

    def _fit_curve(self, kdist_vals, degree=15):
        n   = len(kdist_vals)
        x   = np.linspace(0, 1, n)
        deg = min(degree, n - 1)
        coeffs   = np.polyfit(x, kdist_vals, deg)
        y_fitted = np.polyval(coeffs, x)
        return x, y_fitted, coeffs
    
    def _compute_eps_upper_bound(self):
        idx = np.triu_indices(self.n_features, k=1)
        all_dists = self.D[idx]
        return float(np.percentile(all_dists, 50))

    def _max_curvature_point(self, x, y_fitted, coeffs):
        d1_coeffs = np.polyder(coeffs, 1)
        d2_coeffs = np.polyder(coeffs, 2)

        dy  = np.polyval(d1_coeffs, x)
        d2y = np.polyval(d2_coeffs, x)

        curvature = np.abs(d2y) / (1 + dy**2) ** 1.5

        # Paper: "sudden change region after the curve rises steadily"
        # = upper 70% of the y range within the valid eps ceiling
        eps_ceil  = self._eps_upper_bound
        ceil_mask = y_fitted <= eps_ceil

        y_thresh   = y_fitted[ceil_mask].min() + 0.3 * (
                    y_fitted[ceil_mask].max() - y_fitted[ceil_mask].min()
                    ) if ceil_mask.sum() > 0 else y_fitted.min()

        valid_mask = (y_fitted >= y_thresh) & ceil_mask

        if valid_mask.sum() == 0:
            # Case A: rising region and ceiling don't overlap
            # → drop the "upper 70%" constraint, keep only ceiling
            valid_mask = ceil_mask

        if valid_mask.sum() == 0:
            # Case B: ceiling itself is below all fitted values (degenerate)
            # → return ceiling as eps directly (most conservative valid value)
            return float(eps_ceil)

        curvature_masked = curvature.copy()
        curvature_masked[~valid_mask] = -np.inf
        best_idx = int(np.argmax(curvature_masked))

        return float(y_fitted[best_idx])

    def _generate_eps_list(self, k_max):
        eps_list = []
        for k in range(1, k_max + 1):
            kdist_vals      = self._kdist_curve(k)
            x, y_fitted, coeffs = self._fit_curve(kdist_vals)
            eps_k           = self._max_curvature_point(x, y_fitted, coeffs)
            eps_list.append(eps_k)
        return eps_list

    def _generate_minpts_list(self, eps_list, beta=0.8):
        minpts_list = []
        n = self.n_features
        for eps_k in eps_list:
            neighbor_counts = np.array([ np.sum(self.D[i] <= eps_k) - 1  for i in range(n)])
            expected_count = neighbor_counts.mean()          
            minpts_k       = max(2, int(beta * expected_count))
            minpts_list.append(minpts_k)
        return minpts_list

    def _find_optimal_k(self, eps_list, minpts_list, y_stable=5):
        """
        Paper Section 3.2.3:
        Run DBSCAN for each (Eps_K, MinPts_K) pair. Record cluster count.
        Find the longest run of Y consecutive identical cluster counts —
        that is the stable interval. The maximum K in that interval is
        the optimal K. If no run of length Y exists, try Y-1, Y-2 down to 3.
        If still none, use the interval where cluster count fluctuates ≤ 1.
        """
        cluster_counts = []
        all_labels     = []

        for eps_k, minpts_k in zip(eps_list, minpts_list):
            labels = DBSCAN( eps=eps_k, min_samples=minpts_k, metric="precomputed").fit_predict(self.D)
            n_clusters = len(set(labels) - {-1})
            cluster_counts.append(n_clusters)
            all_labels.append(labels)

        # Find stable interval: Y consecutive identical cluster counts
        optimal_k_idx = None
        for y in range(y_stable, 2, -1):
            for i in range(len(cluster_counts) - y + 1):
                window = cluster_counts[i:i + y]
                if len(set(window)) == 1 and window[0] >= 2:
                    # stable interval found — take the MAX K in it
                    # (i + y - 1 is the last index of the stable run)
                    optimal_k_idx = i + y - 1
                    # keep scanning for a later stable interval at same count
            if optimal_k_idx is not None:
                break

        # Fallback: fluctuation ≤ 1 within a window
        if optimal_k_idx is None:
            for i in range(len(cluster_counts) - 3 + 1):
                window = cluster_counts[i:i + 3]
                if max(window) - min(window) <= 1 and max(window) >= 2:
                    optimal_k_idx = i + 2
                    break

        # Last resort: pick K with most common non-trivial cluster count
        if optimal_k_idx is None:
            valid = [(i, c) for i, c in enumerate(cluster_counts) if c >= 2]
            if valid:
                from collections import Counter
                count_freq   = Counter(c for _, c in valid)
                best_count   = count_freq.most_common(1)[0][0]
                candidates   = [i for i, c in valid if c == best_count]
                optimal_k_idx = max(candidates)   # maximum K in stable mode
            else:
                optimal_k_idx = len(cluster_counts) // 2

        return optimal_k_idx, cluster_counts, all_labels


    def _silhouette_verify(self, optimal_k_idx, eps_list, minpts_list,
                           cluster_counts):
        """
        Paper Section 3.2.4 and Eq. (8):
        S(i) = (b(i) - a(i)) / max(a(i), b(i))
        Compute silhouette for all valid K (cluster count > 1) and confirm
        the chosen K has a high (not necessarily maximum) silhouette.
        Returns silhouette at optimal K for logging.
        """
        from sklearn.metrics import silhouette_score
        results = []
        for i, (eps_k, minpts_k) in enumerate(zip(eps_list, minpts_list)):
            if cluster_counts[i] < 2:
                results.append(np.nan)
                continue
            labels = DBSCAN(
                eps=eps_k, min_samples=minpts_k, metric="precomputed"
            ).fit_predict(self.D)
            # silhouette needs at least 2 clusters and non-noise points
            mask = labels != -1
            if mask.sum() < 2 or len(set(labels[mask])) < 2:
                results.append(np.nan)
                continue
            try:
                s = silhouette_score(self.D[np.ix_(mask, mask)],
                                     labels[mask], metric="precomputed")
                results.append(float(s))
            except Exception:
                results.append(np.nan)

        sil_at_optimal = results[optimal_k_idx] if optimal_k_idx < len(results) else np.nan
        return sil_at_optimal, results

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #
    def cluster_features(self, feature_names, beta=0.8, y_stable=5, verify=True):
        n      = self.n_features
        k_max  = n - 1    
        self._eps_upper_bound = self._compute_eps_upper_bound()
        print(f"Data-driven eps ceiling: {self._eps_upper_bound:.4f}")
        print(f"Running X-DBSCAN over K = 1..{k_max}")

        
        print("  Generating Eps list (polynomial fit + max curvature)...")
        self._eps_list    = self._generate_eps_list(k_max)

        print("  Generating MinPts list (expectation + noise threshold)...")
        self._minpts_list = self._generate_minpts_list(self._eps_list, beta=beta)

        
        print("  Finding stable interval of cluster counts...")
        opt_idx, cluster_counts, all_labels = self._find_optimal_k( self._eps_list, self._minpts_list, y_stable=y_stable)

        self.optimal_k   = opt_idx + 1    # K is 1-indexed
        self.eps         = self._eps_list[opt_idx]
        self.min_samples = self._minpts_list[opt_idx]
        labels           = all_labels[opt_idx]

        # Reconstruct clusters and noise from optimal labels
        unique_labels  = set(labels) - {-1}
        self.clusters  = {cid: [] for cid in unique_labels}
        self.noise     = []
        for idx, label in enumerate(labels):
            if label == -1:
                self.noise.append(idx)
            else:
                self.clusters[label].append(idx)

        self.n_clusters = len(self.clusters)
        self.noise_pct  = len(self.noise) / self.n_features * 100

        # Step 5 (optional): silhouette verification
        sil = None
        if verify:
            print("  Verifying via silhouette coefficient...")
            sil, _ = self._silhouette_verify(
                opt_idx, self._eps_list, self._minpts_list, cluster_counts
            )

        print(f"\nOptimal K       : {self.optimal_k}")
        print(f"Optimal Eps     : {self.eps:.4f}")
        print(f"Optimal MinPts  : {self.min_samples}")
        print(f"Clusters found  : {self.n_clusters}")
        print(f"Noise features  : {len(self.noise)}  ({self.noise_pct:.1f}%)")
        if sil is not None:
            print(f"Silhouette      : {sil:.4f}")

        self._plot_cluster_stability(cluster_counts)
        return self

    # ------------------------------------------------------------------ #
    #  Noise filtering — MI screen before passing downstream             #
    # ------------------------------------------------------------------ #
    def filter_noise(self, X, y, mi_threshold=0.01):
        """
        Screen noise features by MI with target.
        DBSCAN noise = structurally unique w.r.t. correlation structure,
        but may still be irrelevant to the target. MI filter removes
        features with no target association (Battiti, 1994).
        random_state=0 makes the sklearn MI estimator effectively stable
        for thresholding purposes.
        """
        from sklearn.feature_selection import mutual_info_classif
        if not self.noise:
            return []
        noise_X   = X[:, self.noise]
        mi_scores = mutual_info_classif(
            noise_X, y, discrete_features=False, random_state=0
        )
        kept    = [self.noise[i] for i, s in enumerate(mi_scores)
                   if s >= mi_threshold]
        dropped = len(self.noise) - len(kept)
        print(f"  Noise MI filter: kept {len(kept)}, "
              f"dropped {dropped} (MI < {mi_threshold})")
        return kept

    # ------------------------------------------------------------------ #
    #  Plots                                                              #
    # ------------------------------------------------------------------ #
    def _plot_cluster_stability(self, cluster_counts):
        """Plot cluster count vs K — shows the stable interval visually."""
        plt.figure(figsize=(8, 3))
        plt.plot(range(1, len(cluster_counts) + 1), cluster_counts,
                 color="steelblue", linewidth=1.5)
        plt.axvline(self.optimal_k, color="orange", linestyle="--",
                    linewidth=1.2, label=f"Optimal K={self.optimal_k}")
        plt.xlabel("K value")
        plt.ylabel("Number of clusters")
        plt.title("Cluster count stability across K values (X-DBSCAN)")
        plt.legend()
        plt.tight_layout()
        plt.show()



def main():
    
    X, y = make_classification(
        n_samples=1000, n_features=30,
        n_informative=15, n_redundant=10,
        n_repeated=0, n_classes=2, random_state=42
    )
    feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]

    model = XDBSCAN_Clustering()
    model.build_distance_matrix(X, use_spearman=False)
    model.cluster_features(feature_names=feature_cols, verify=True)

    print("\nCluster assignments:")
    for cid, idxs in model.clusters.items():
        names = [feature_cols[i] for i in idxs]
        print(f"  Cluster {cid}: {names}")
    if model.noise:
        print(f"  Noise: {[feature_cols[i] for i in model.noise]}")

if __name__ == "__main__":
    main()