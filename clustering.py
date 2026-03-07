import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from scipy.stats import spearmanr


class DBSCAN_Clustering():
    def __init__(self):
        self.D = None
        self.n_features=0
        self.min_samples = 2
        self.eps = 0
        self.n_clusters=0
        self.noise_pct=0

        self.clusters={}
        self.noise=[]
        
    def build_distance_matrix(self, data, use_spearman=False):
        if use_spearman:
            corr_matrix, _ = spearmanr(data, axis=0)
            corr_matrix = np.array(corr_matrix)
        else:
            corr_matrix = np.corrcoef(data.T)

        self.D = 1.0 - np.abs(corr_matrix)
        np.fill_diagonal(self.D, 0.0)
        self.D = np.clip(self.D, 0.0, 1.0)
        self.n_features=self.D.shape[0]

    def knn_distances(self):
        n = self.D.shape[0]
        knn_dists = np.zeros(n)
        for i in range(n):
            row = np.sort(self.D[i])  
            knn_dists[i] = row[min(self.min_samples, n - 1)]
        return np.sort(knn_dists)
    
    def detect_elbow(self, sorted_dists): 
        n = len(sorted_dists)
        x = np.arange(n, dtype=float) / (n - 1)
        y = (sorted_dists - sorted_dists.min()) / (sorted_dists.max() - sorted_dists.min())
       
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        dx, dy = x1 - x0, y1 - y0
        line_len = np.sqrt(dx**2 + dy**2)

        if line_len < 1e-12:
            return float(np.median(sorted_dists))

        """
        dist= |Ax+By+C|/root(A^2 + B^2)
        A=dy
        B=-dx
        C= -(dy*x0-dx*y0)"""
        perp = np.abs(dy * x - dx * y - (dy * x0 - dx * y0)) / line_len
        best_idx = np.argmax(perp)
        
        return float(sorted_dists[best_idx])

   
    def plot_kdist(self, knn_dists):
        plt.figure(figsize=(8, 3))
        plt.plot(knn_dists, color="steelblue", linewidth=1.8, label="k-NN distance")
        plt.axhline(self.eps, color="orange", linestyle=":", linewidth=1.2,
                    label=f"auto-elbow = {self.eps:.4f}")
        plt.xlabel("Features (sorted by NN distance)")
        plt.ylabel("NN Distance")
        plt.title("K-Distance Plot for eps Selection")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sanity_check(self):
        if self.n_clusters == 0 or self.n_clusters == 1:
            self.eps -= 0.05
            self.eps = max(self.eps, 1e-3)
            print("eps decreased to ",self.eps)
            return False 

        if self.noise_pct > 60:
            self.eps += 0.05
            self.eps=min(self.eps,0.99)
            print("eps increased to: ",self.eps)
            return False  
        return True
    
    def cluster_features(self, feature_names,max_iter=20):
        knn_dists = self.knn_distances()
        self.eps = self.detect_elbow(knn_dists)
        print("eps calculated through elbow curve: ", self.eps)
        self.plot_kdist(knn_dists)

        for _ in range(max_iter):
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
            labels = db.fit_predict(self.D)

            unique_labels = set(labels) - {-1}
            self.clusters = {cid: [] for cid in unique_labels}
            self.noise=[]

            for idx, label in enumerate(labels):
                if label == -1:
                    self.noise.append(idx)
                else:
                    self.clusters[label].append(idx)

            self.n_clusters = len(self.clusters)
            self.noise_pct = len(self.noise) / self.n_features * 100
            
            print(f"Clusters found : {self.n_clusters}")
            print(f"Noise features : {len(self.noise)}  ({self.noise_pct:.1f}%)")
            
            if self.sanity_check():
                break
 
           




def main():
    def generate_dataset():
        X, y = make_classification(
            n_samples=1000,
            n_features=30,
            n_informative=15,
            n_redundant=10,
            n_repeated=0,
            n_classes=2,
            random_state=42
        )
        feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y
        return df

    df = generate_dataset()

 
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values

    model = DBSCAN_Clustering()
    model.build_distance_matrix(X, use_spearman=False)
   

    model.cluster_features(feature_names=feature_cols)

    print("\nCluster assignments:")
    for cid, idxs in model.clusters.items():
        names = [feature_cols[i] for i in idxs]
        print(f"  Cluster {cid}: {names}")
    if model.noise:
        noise_names = [feature_cols[i] for i in model.noise]
        print(f"  Noise features: {noise_names}")


if __name__ == "__main__":
    main()