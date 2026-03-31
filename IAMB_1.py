from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from clustering import DBSCAN_Clustering
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class IAMB:
    def __init__(self):
        self.n_bins        = None   
        self.G             = []     
        self.cmi_threshold = None   

    def discretize(self, Z):
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        kbd = KBinsDiscretizer(
            n_bins=self.n_bins, encode='ordinal', strategy='uniform'
        )
        return kbd.fit_transform(Z).astype(int)

   
    def _mi_hist(self, x, y):
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

   
    def cmi(self, x, y, Z):
        Z_disc = self.discretize(Z)
        if Z_disc.shape[1] == 1:
            strata = Z_disc[:, 0]
        else:
            _, strata = np.unique(Z_disc, axis=0, return_inverse=True)

        n            = len(x)
        cmi_value    = 0.0
        total_p_used = 0.0

        for stratum in np.unique(strata):
            mask = strata == stratum
            n_z  = mask.sum()
            if n_z < 5:
                continue
            p_z           = n_z / n
            mi            = self._mi_hist(x[mask], y[mask])
            cmi_value    += p_z * mi
            total_p_used += p_z

        # rescale by used probability mass to avoid deflation
        if total_p_used > 0:
            cmi_value /= total_p_used

        return cmi_value


    def _build_global_context(self, X, y, cap=15):
        n_features = X.shape[1]
        mi_all = np.array([self._mi_hist(X[:, f], y) for f in range(n_features)])

        mean_mi = np.mean(mi_all)
        median_mi    = np.median(mi_all)
        G_candidates = [f for f in range(n_features) if mi_all[f] >= mean_mi]
        G_candidates.sort(key=lambda f: mi_all[f], reverse=True)
        G = G_candidates[:cap]
        self.G=G
        self.cmi_threshold = 0.05 * (mi_all.max() + 1e-12)

        print(f"\n[Global MI]  median={median_mi:.4f}  mean={mean_mi:.4f} "
              f"|G|={len(G)}  cmi_threshold={self.cmi_threshold:.4f}")
        print(f"  G = {G}")
        return mi_all, G

  
    def _forward_with_context(self, target, cluster_indices, X_full, context):
       
        candidates = list(cluster_indices)   
        MB         = []                      

        while True:
            best_feature, best_score = None, -1.0

            for feature in candidates:
                if feature in MB:
                    continue

                x         = X_full[:, feature]
                z_indices = MB + context       

                if len(z_indices) == 0:
                    score = self._mi_hist(x, target)
                else:
                    score = self.cmi(x, target, X_full[:, z_indices])

                if score > best_score:
                    best_score   = score
                    best_feature = feature

            if best_feature is None or best_score < self.cmi_threshold:
                break

            MB.append(best_feature)
            print(f"    + feat={best_feature}  CMI={best_score:.4f}")

        return MB


    def _add_weak_features(self, cluster_indices, MB, mi_all, weak_frac=0.30):
        
        not_selected = [f for f in cluster_indices if f not in MB]
        if not not_selected:
            return MB

        coverage = len(MB) / len(cluster_indices)
        if coverage >= 0.5:
            return MB

        n_weak = max(1, int(np.ceil(len(not_selected) * weak_frac)))
        to_add = sorted(not_selected,
                        key=lambda f: mi_all[f], reverse=True)[:n_weak]

        for f in to_add:
            MB.append(f)
            print(f"    ~ weak feat={f}  MI={mi_all[f]:.4f}")

        return MB

   
    def _run_cluster(self, cid, cluster_indices, X_full, y, mi_all):
        print(f"\n[Cluster {cid}]  features={cluster_indices}")

        if len(cluster_indices) == 1:
            print(f"  Single-feature cluster, kept directly.")
            return list(cluster_indices)

        context = [f for f in self.G if f not in cluster_indices]
        print(f"  context (G\\cluster) = {context}")

       
        MB = self._forward_with_context(
            target          = y,
            cluster_indices = cluster_indices,
            X_full          = X_full,
            context         = context,
        )

      
        if len(MB) == 0:
            best = max(cluster_indices, key=lambda f: mi_all[f])
            MB   = [best]
            print(f"  MB empty — fallback kept feat={best}  "
                  f"MI={mi_all[best]:.4f}")

      
        MB = self._add_weak_features(cluster_indices, MB, mi_all)

        print(f"  → selected {len(MB)}/{len(cluster_indices)}: {MB}")
        return MB

    def run(self, clusters, noise_indices, X, y, context_cap=5):
        
        self.n_bins = max(5, min(20, round(X.shape[0] ** (1 / 3))))
        mi_all, self.G = self._build_global_context(X, y, cap=context_cap)
        selected = []

        for cid, indices in clusters.items():
            chosen = self._run_cluster(cid, indices, X, y, mi_all)
            selected.extend(chosen)
            
        selected.extend(self.G)

        if noise_indices:
            noise_mi     = {f: mi_all[f] for f in noise_indices}
            median_noise = np.median(list(noise_mi.values()))
            retained     = [f for f, v in noise_mi.items() if v >= median_noise]
            print(f"\n[Noise]  retained {len(retained)}/{len(noise_indices)} "
                  f"(MI >= {median_noise:.4f}): {retained}")
            selected.extend(retained)
            
        print(noise_indices)

        selected = sorted(set(selected))
        print(f"\n[IAMB done]  |S_reduced| = {len(selected)}  {selected}")
        return selected

def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=10,
        n_redundant=20,
        n_repeated=5,
        n_classes=2,
        random_state=42,
        shuffle=False
    )
    feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DBSCAN_Clustering()
    model.build_distance_matrix(X_train)
    model.cluster_features(feature_names=feature_cols)

    print("\nCluster assignments:")
    for cid, idxs in model.clusters.items():
        names = [feature_cols[i] for i in idxs]
        print(f"  Cluster {cid}: {names}")
    if model.noise:
        noise_names = [feature_cols[i] for i in model.noise]
        print(f"  Noise features: {noise_names}")

    iamb = IAMB()
    selected = iamb.run(
        clusters      = model.clusters,
        noise_indices = model.noise,
        X             = X_train,
        y             = y_train,
        context_cap   = 25,
    )

    selected_names = [feature_cols[i] for i in selected]
    print("\nSelected indices :", selected)
    print("Selected features:", selected_names)
    print("Count            :", len(selected_names))


if __name__ == "__main__":
    main()