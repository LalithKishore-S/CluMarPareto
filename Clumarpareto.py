from clustering import DBSCAN_Clustering
from X_DBSCAN import XDBSCAN_Clustering
from IAMB import IAMB
from HITONMB import HITONMB
from NSGA2 import NSGA2_FS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CluMarPareto:
    """
    Unified pipeline:
        DBSCAN  →  IAMB  →  NSGA2
 
    Parameters
    ──────────
    classifier      : 'decisiontree' | 'randomforest'
    population_size : NSGA2 population size
    n_generations   : NSGA2 generation cap
    crossover_rate  : probability of single-point crossover
    base_alpha      : IAMB base significance level (auto-scaled per cluster)
    n_bins          : bins for CMI discretisation in IAMB
    min_samples     : DBSCAN min_samples parameter
    use_spearman    : use Spearman instead of Pearson correlation in DBSCAN
    test_size       : fraction of data held out so IAMB runs on train split only
    verbose         : print progress
    """
 
    def __init__(
        self,
        classifier:      str   = "decisiontree",
        crossover_rate:  float = 0.80,
        verbose:         bool  = True,
    ):
        self.classifier      = classifier
        self.crossover_rate  = crossover_rate
        self.verbose         = verbose
 
        self.feature_names_       = None   # all feature names (excl. target)
        self.selected_indices_    = None   # indices in original feature space
        self.selected_features_   = None   # feature names
        self.knee_solution_       = None   # Individual at knee
        self.pareto_front_        = None   # full Pareto front
 
        self.dbscan_ = None
        self.iamb_   = None
        self.nsga2_  = None
 
    def fit(self, data: pd.DataFrame) -> "CluMarPareto":
        # pass training data alone
        self.feature_names_ = list(data.columns)[:-1]
        X = data[self.feature_names_].values
        y = data[data.columns[-1]].values

        if self.verbose:
            print("=" * 60)
            print("STAGE 1 — DBSCAN Clustering")
            print("=" * 60)
 
        self.dbscan_ = XDBSCAN_Clustering()
        self.dbscan_.build_distance_matrix(X)
        self.dbscan_.cluster_features(feature_names=self.feature_names_)
 

        if self.verbose:
            print("\n" + "=" * 60)
            print("STAGE 2 — IAMB within clusters")
            print("=" * 60)
 
        self.iamb_ = IAMB()
        self.selected_indices_ = self.iamb_.run(clusters=self.dbscan_.clusters, X=X, y=y)
        # informative_noise = self.dbscan_.filter_noise(X, y)
        informative_noise = self.dbscan_.noise
        self.selected_indices_ += informative_noise
        self.selected_features_ = [self.feature_names_[i] for i in self.selected_indices_]
 
        if self.verbose:
            reduction = (1 - len(self.selected_indices_) /
                         len(self.feature_names_)) * 100
            print(f"\n  Original features : {len(self.feature_names_)}")
            print(f"  After IAMB        : {len(self.selected_indices_)} "
                  f"({reduction:.1f}% reduction)")
            print(f"  Features passed to NSGA2: {self.selected_features_}")
 
        if self.verbose:
            print("\n" + "=" * 60)
            print("STAGE 3 — NSGA2 on reduced feature space")
            print("=" * 60)
 
        
        X_reduced = X[:, self.selected_indices_]
        self.population_size = min(200, max(50, 10 * X_reduced.shape[1]))
        self.n_generations = min(200, max(50, self.population_size // 2))
        warm_start_local = list(range(len(self.selected_indices_)))

 
        self.nsga2_ = NSGA2_FS(
            classifier=self.classifier,
            population_size=self.population_size,
            n_generations=self.n_generations,
            crossover_rate=self.crossover_rate,
        )
        self.nsga2_.fit(X=X_reduced, y=y, warm_start_indices = warm_start_local)
        self.pareto_front_ = self.nsga2_.pareto_front_
        self.knee_solution_ = self.nsga2_.find_knee_point(self.pareto_front_)
 
        # map knee mask back to original feature indices
        knee_local_indices    = np.where(self.knee_solution_.mask_features)[0]
        knee_original_indices = [self.selected_indices_[i] for i in knee_local_indices]
        knee_feature_names    = [self.feature_names_[i] for i in knee_original_indices]
 
        # if self.verbose:
        #     print("\n" + "=" * 60)
        #     print("RESULT — Knee-point solution")
        #     print("=" * 60)
        #     print(f"  Features selected : {self.knee_solution_.obj_scores[0]}")
        #     print(f"  Accuracy          : {self.knee_solution_.obj_scores[1]:.4f}")
        #     print(f"  Feature names     : {knee_feature_names}")
 
        self.knee_original_indices_ = knee_original_indices
        self.knee_feature_names_    = knee_feature_names

        if self.verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"  Original features  ({len(self.feature_names_)}):")
            print(f"    {self.feature_names_}")
            print(f"\n  After IAMB         ({len(self.selected_features_)}):")
            print(f"    {self.selected_features_}")
            print(f"\n  After NSGA2 / Knee ({len(self.knee_feature_names_)}):")
            print(f"    {self.knee_feature_names_}")
            print(f"\n  Accuracy (CV)      : {self.knee_solution_.obj_scores[1]:.4f}")
            print(f"  Features reduced   : {len(self.feature_names_)} → "
                f"{len(self.selected_features_)} → "
                f"{len(self.knee_feature_names_)}")
            print(f"  Total reduction    : "
                f"{(1 - len(self.knee_feature_names_) / len(self.feature_names_)) * 100:.1f}%")
        return self
 
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return X filtered to knee-point features (original index space)."""
        return X[:, self.knee_original_indices_]
 
    def transform_iamb(self, X: np.ndarray) -> np.ndarray:
        """Return X filtered to all IAMB-selected features."""
        return X[:, self.selected_indices_]
 
 