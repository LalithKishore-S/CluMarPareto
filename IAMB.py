from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif  
import numpy as np
from clustering import DBSCAN_Clustering
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class IAMB:
    def __init__(self):
        self.alpha = None
        self.n_bins = None
           
    def discretize(self, Z):
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        return kbd.fit_transform(Z).astype(int)

    def cmi(self, x, y, Z):
        Z_disc = self.discretize(Z)
        if Z_disc.shape[1] == 1:
            strata = Z_disc[:, 0]
        else:
            _, strata = np.unique(Z_disc, axis=0, return_inverse=True)
        n = len(x)
        cmi_value = 0.0

        for stratum in np.unique(strata):
            mask = strata == stratum
            n_z = mask.sum()
            if n_z < 5:
                continue
            p_z = n_z / n            
            mi = mutual_info_classif(
                x[mask].reshape(-1, 1),
                y[mask],
                discrete_features=False
            )[0]
            cmi_value += p_z * mi
        return cmi_value
    
    def _mi(self, x, y):
        return mutual_info_classif(
            x.reshape(-1, 1),
            y,
            discrete_features=False
        )[0]

    def forward_phase(self, target, candidates, X_cluster, MB):
        while True:
            best_feature, best_cmi = None, -1

            for feature in candidates:
                if feature in MB:
                    continue
                x = X_cluster[:, feature]
                if len(MB) == 0:
                    score = self._mi(x, target)
                else:
                    Z = X_cluster[:, list(MB)]
                    score = self.cmi(x, target, Z)
                if score > best_cmi:
                    best_cmi = score
                    best_feature = feature
            if best_feature is None or best_cmi <= self.alpha:
                break   

            MB.append(best_feature)
            print(f"Forward phase added feature {best_feature}  CMI={best_cmi:.4f}")
        return MB

    def backward_phase(self, target, X_cluster, MB):
        to_remove = []
        for feature in reversed(list(MB)):
            rest = [m for m in MB if m != feature]
            x = X_cluster[:, feature]

            if len(rest) == 0:
                continue

            Z = X_cluster[:, rest]
            score = self.cmi(x, target, Z)

            if score <= self.alpha:
                to_remove.append(feature)
                print(f"Backward phase removed feature {feature}  CMI={score:.4f}")

        for feature in to_remove:
            MB.remove(feature)
        return MB

    def iamb(self, cluster_indices, X_train, y_train):
        if len(cluster_indices) == 1:
            print(f"Single feature {cluster_indices[0]}")
            return cluster_indices

        print(cluster_indices)
        X_cluster = X_train[:, cluster_indices]
        local_candidates = list(range(X_cluster.shape[1]))
        MB = []

        MB = self.forward_phase(y_train, local_candidates, X_cluster, MB)
        MB = self.backward_phase(y_train, X_cluster, MB)

        if len(MB) == 0:
            mi_scores = [self._mi(X_cluster[:, f], y_train) for f in local_candidates]
            best_local = int(np.argmax(mi_scores))
            MB = [best_local]
            print(f"Fallback MB is empty,so kept highest MI: "
                  f"global idx {cluster_indices[best_local]}")

        selected_global = [cluster_indices[i] for i in MB]
        return selected_global

    def run(self, clusters, X, y):
        self.alpha = 0.05 / X.shape[1]
        self.n_bins = max(5, min(20, round(X.shape[0] ** (1/3))))
        selected_from_clusters = []

        for cid, indices in clusters.items():
            print(f"\n[IAMB] Cluster {cid} — features {indices}")
            selected = self.iamb(indices, X, y)
            print(f"  → Selected: {selected}")
            selected_from_clusters.extend(selected)

        return selected_from_clusters
    
def main():
    
   
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=12,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )
    feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

   
    model = DBSCAN_Clustering()
    model.build_distance_matrix(X)          
    model.cluster_features(feature_names=feature_cols)

    print("\nCluster assignments:")
    for cid, idxs in model.clusters.items():
        names = [feature_cols[i] for i in idxs]
        print(f"  Cluster {cid}: {names}")
    if model.noise:
        noise_names = [feature_cols[i] for i in model.noise]
        print(f"  Noise features: {noise_names}")


    iamb = IAMB(alpha=0.001, n_bins=10)
    selected = iamb.run(model.clusters, X_train,y_train)
    selected_feature_names=[feature_cols[i] for i in selected]
    print("\nSelected from clusters:", selected)
    print("Selected feature names:", selected_feature_names)
    print("length of selected features: ",len(selected_feature_names))
    
if __name__ == "__main__":
    main()