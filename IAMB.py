from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif  
import numpy as np
from clustering import DBSCAN_Clustering
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

class IAMB:
    def __init__(self):
        self.n_bins = None
           
    def discretize(self, Z):
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        return kbd.fit_transform(Z).astype(int)

    # def cmi(self, x, y, Z):
    #     Z_disc = self.discretize(Z)
    #     if Z_disc.shape[1] == 1:
    #         strata = Z_disc[:, 0]
    #     else:
    #         _, strata = np.unique(Z_disc, axis=0, return_inverse=True)
    #     n = len(x)
    #     cmi_value = 0.0

    #     for stratum in np.unique(strata):
    #         mask = strata == stratum
    #         n_z = mask.sum()
    #         if n_z < 5:
    #             continue
    #         p_z = n_z / n            
    #         mi = mutual_info_classif(
    #             x[mask].reshape(-1, 1),
    #             y[mask],
    #             discrete_features=False
    #         )[0]
    #         cmi_value += p_z * mi
    #     return cmi_value
    def cmi(self, x, y, Z):
        Z_disc = self.discretize(Z)
        if Z_disc.shape[1] == 1:
            strata = Z_disc[:, 0]
        else:
            _, strata = np.unique(Z_disc, axis=0, return_inverse=True)
        n = len(x)
        cmi_value    = 0.0
        total_p_used = 0.0                          # ← track used mass

        for stratum in np.unique(strata):
            mask = strata == stratum
            n_z = mask.sum()
            if n_z < 5:
                continue
            p_z        = n_z / n
            mi         = self._mi_hist(x[mask], y[mask])   # ← histogram, not sklearn
            cmi_value    += p_z * mi
            total_p_used += p_z                     # ← accumulate used mass

        if total_p_used > 0:
            cmi_value /= total_p_used               # ← rescale to avoid deflation

        return cmi_value
    
    def _mi_hist(self, x, y):
        """
        Marginal MI via histogram — same estimator as cmi().
        Used when MB=∅ so scores are on the same scale throughout.
        Paper: I(X; T | CMB=∅) reduces to I(X; T).
        Scott (1979): n_bins set by cube-root rule in run().
        """
        x_bins = np.linspace(x.min(), x.max() + 1e-10, self.n_bins + 1)
        x_disc = np.digitize(x, x_bins[1:-1])
        mi = 0.0
        for xi in np.unique(x_disc):
            for yi in np.unique(y):
                pxy = np.mean((x_disc == xi) & (y == yi))
                px  = np.mean(x_disc == xi)
                py  = np.mean(y == yi)
                if pxy > 0:
                    mi += pxy * np.log(pxy / (px * py + 1e-12))
        return max(0.0, mi)
    
    def _independence_test(self, x, y, Z, n_samples, alpha):
        """
        I(X; T | Z): returns True if X is INDEPENDENT of T given Z.

        Test statistic: G = 2 * n * CMI(X; T | Z)
        Under H₀ (independence): G ~ chi-square(df)
        df = (rx - 1)(ry - 1) * rz

        rx = distinct bins of x, ry = distinct classes of y,
        rz = distinct strata of Z (1 if Z is None).

        Reference: Tsamardinos et al. (FLAIRS 2003) — asymptotic
        chi-square test for CMI-based conditional independence.

        Returns:
            True  → independent → do NOT add (forward) / remove (backward)
            False → dependent   → add (forward) / keep (backward)
        """
        ry     = len(np.unique(y))
        x_bins = np.linspace(x.min(), x.max() + 1e-10, self.n_bins + 1)
        x_disc = np.digitize(x, x_bins[1:-1])
        rx     = len(np.unique(x_disc))

        if Z is None:
            rz      = 1
            cmi_val = self._mi_hist(x, y)
        else:
            Z_disc = self.discretize(Z)
            if Z_disc.shape[1] == 1:
                strata = Z_disc[:, 0]
            else:
                _, strata = np.unique(Z_disc, axis=0, return_inverse=True)
            rz      = len(np.unique(strata))
            cmi_val = self.cmi(x, y, Z)

        df = (rx - 1) * (ry - 1) * rz
        if df <= 0:
            return True   # degenerate — treat as independent

        G       = 2 * n_samples * cmi_val
        p_value = 1 - chi2.cdf(G, df=df)
        return p_value > alpha   # True = independent

    def forward_phase(self, target, candidates, X_cluster, MB, alpha):
        n = len(target)
        while True:
            best_feature, best_cmi = None, -1

            for feature in candidates:
                if feature in MB:
                    continue
                x = X_cluster[:, feature]
                if len(MB) == 0:
                    score = self._mi_hist(x, target)
                else:
                    Z = X_cluster[:, list(MB)]
                    score = self.cmi(x, target, Z)
                if score > best_cmi:
                    best_cmi = score
                    best_feature = feature
            if best_feature is None :
                break   
            x = X_cluster[:, best_feature]
            Z = X_cluster[:, list(MB)] if len(MB) > 0 else None
            if not self._independence_test(x, target, Z, n, alpha):
                MB.append(best_feature)
                print(f"  Forward added   {best_feature}  CMI={best_cmi:.4f}")
            else:
                break
        return MB

    def backward_phase(self, target, X_cluster, MB, alpha):
        n         = len(target)
        to_remove = []

        for feature in list(MB):
            rest = [m for m in MB if m != feature]
            if len(rest) == 0:
                continue
            x  = X_cluster[:, feature]
            Z  = X_cluster[:, rest]
            if self._independence_test(x, target, Z, n, alpha):
                to_remove.append(feature)
                print(f"  Backward removed {feature:>3}  "
                      f"(independent given CMB\\{{{feature}}})")

        for f in to_remove:
            MB.remove(f)

        return MB

    def iamb(self, cluster_indices, X_train, y_train, alpha_local):
        if len(cluster_indices) == 1:
            print(f"Single feature {cluster_indices[0]}")
            return cluster_indices

        print(cluster_indices)
        X_cluster = X_train[:, cluster_indices]
        local_candidates = list(range(X_cluster.shape[1]))
        MB = []

        while True:
            MB_before = list(MB)

            # Forward: add dependent features
            MB = self.forward_phase(y_train, local_candidates, X_cluster, MB, alpha_local)
            MB = self.backward_phase(y_train, X_cluster, MB, alpha_local)

            # Stop when neither phase changed MB
            if set(MB) == set(MB_before):
                break

        if len(MB) == 0:
            mi_scores = [self._mi_hist(X_cluster[:, f], y_train) for f in local_candidates]
            n_keep    = max(1, int(np.ceil(len(cluster_indices) / 3)))
            top_local = sorted(range(len(mi_scores)),
                            key=lambda i: mi_scores[i], reverse=True)[:n_keep]
            MB = top_local
            print(f"  Fallback: kept top {n_keep} of {len(cluster_indices)} by MI")

        selected_global = [cluster_indices[i] for i in MB]
        return selected_global

    def run(self, clusters, X, y):
        # self.alpha = 0.05 / X.shape[1]
        self.n_bins = max(5, min(20, round(X.shape[0] ** (1/3))))
        selected_from_clusters = []

        for cid, indices in clusters.items():
            alpha_local = 0.05 / len(indices)
            print(f"\n[IAMB] Cluster {cid} — features {indices}")
            selected = self.iamb(indices, X, y, alpha_local)
            print(f"  → Selected: {selected}")
            selected_from_clusters.extend(selected)

        return selected_from_clusters
    
# def main():
    
   
#     X, y = make_classification(
#         n_samples=1000,
#         n_features=30,
#         n_informative=12,
#         n_redundant=10,
#         n_repeated=0,
#         n_classes=2,
#         random_state=42
#     )
#     feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]

   
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

   
#     model = DBSCAN_Clustering()
#     model.build_distance_matrix(X)          
#     model.cluster_features(feature_names=feature_cols)

#     print("\nCluster assignments:")
#     for cid, idxs in model.clusters.items():
#         names = [feature_cols[i] for i in idxs]
#         print(f"  Cluster {cid}: {names}")
#     if model.noise:
#         noise_names = [feature_cols[i] for i in model.noise]
#         print(f"  Noise features: {noise_names}")


#     iamb = IAMB(alpha=0.001, n_bins=10)
#     selected = iamb.run(model.clusters, X_train,y_train)
#     selected_feature_names=[feature_cols[i] for i in selected]
#     print("\nSelected from clusters:", selected)
#     print("Selected feature names:", selected_feature_names)
#     print("length of selected features: ",len(selected_feature_names))
    
# if __name__ == "__main__":
#     main()