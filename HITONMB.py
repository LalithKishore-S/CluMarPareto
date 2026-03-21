"""
IAMB.py  —  HITON-MB implementation
Replaces the original IAMB algorithm with HITON-MB for better quality.

Key improvement over IAMB:
    - Separates PC (Parents & Children) discovery from Spouse discovery
    - Backward phase uses small subset conditioning instead of full MB
    - Prevents over-conditioning which caused strong features to be removed

Reference:
    Aliferis, C. F., Tsamardinos, I., & Statnikov, A. (2003).
    HITON: A novel Markov Blanket algorithm for optimal variable selection.
    AMIA Annual Symposium Proceedings, 2003, 21–25.
"""

from itertools import combinations
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from clustering import DBSCAN_Clustering


class HITONMB:
    """
    HITON-MB based Markov Blanket discovery.
    Drop-in replacement for the original IAMB class —
    same interface, same inputs, same outputs.

    Hyperparameters are auto-derived in run():
        alpha  = 0.05 / n_features   (Bonferroni correction)
        n_bins = max(5, min(20, round(n_samples^(1/3))))  (cube-root rule)
    """

    def __init__(self):
        self.alpha  = None   # set in run()
        self.n_bins = None   # set in run()

    # ---------------------------------------------------------------- helpers

    def _discretize(self, Z: np.ndarray) -> np.ndarray:
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        kbd = KBinsDiscretizer(
            n_bins=self.n_bins, encode='ordinal', strategy='uniform'
        )
        return kbd.fit_transform(Z).astype(int)

    def _mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Unconditional mutual information I(x ; y)."""
        return float(
            mutual_info_classif(x.reshape(-1, 1), y, discrete_features=False)[0]
        )

    def _cmi(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
        """
        Conditional mutual information I(x ; y | Z).
        Computed by stratifying on discretised Z and taking
        a weighted sum of MI within each stratum.
        """
        Z_disc = self._discretize(Z)
        if Z_disc.shape[1] == 1:
            strata = Z_disc[:, 0]
        else:
            _, strata = np.unique(Z_disc, axis=0, return_inverse=True)

        n, cmi = len(x), 0.0
        for s in np.unique(strata):
            mask = strata == s
            if mask.sum() < 5:
                continue
            cmi += (mask.sum() / n) * self._mi(x[mask], y[mask])
        return cmi

    # ---------------------------------------------------- Phase 1: find PC

    def _subset_removal(
        self,
        X_local: np.ndarray,
        y: np.ndarray,
        PC: list,
    ) -> list:
        """
        After each forward addition, check whether any current PC member
        becomes conditionally independent of the target given a SMALL
        subset (size 1 or 2) of the remaining PC members.

        Using small subsets instead of the full PC prevents the
        over-conditioning that caused IAMB to remove strong features.
        """
        to_remove = []
        for f in list(PC):
            if f in to_remove:
                continue
            rest = [m for m in PC if m != f and m not in to_remove]
            if not rest:
                continue
            # test subsets of size 1 and 2 only
            max_size = min(3, len(rest) + 1)
            removed  = False
            for size in range(1, max_size):
                for subset in combinations(rest, size):
                    score = self._cmi(
                        X_local[:, f], y,
                        X_local[:, list(subset)]
                    )
                    if score <= self.alpha:
                        to_remove.append(f)
                        removed = True
                        break
                if removed:
                    break

        for f in to_remove:
            PC.remove(f)
        return PC

    def _find_PC(self, X_local, y, candidates):
        PC        = []
        blacklist = set()   # ← features removed by subset removal

        while True:
            best_feat, best_score = None, -1.0
            for f in candidates:
                if f in PC or f in blacklist:   # ← skip blacklisted
                    continue
                score = (
                    self._mi(X_local[:, f], y)
                    if not PC
                    else self._cmi(X_local[:, f], y, X_local[:, PC])
                )
                if score > best_score:
                    best_score, best_feat = score, f

            if best_feat is None or best_score <= self.alpha:
                break

            PC.append(best_feat)
            print(f"  [PC] added feature {best_feat}  CMI={best_score:.4f}")

            before   = set(PC)
            PC       = self._subset_removal(X_local, y, PC)
            removed  = before - set(PC)
            blacklist.update(removed)           # ← blacklist removed features

            if removed:
                print(f"  [PC] subset removal kept {len(PC)}, "
                    f"blacklisted {removed}")

        return PC

    # ------------------------------------------------- Phase 2: find Spouses

    def _find_spouses(
        self,
        X_local: np.ndarray,
        y: np.ndarray,
        PC: list,
        candidates: list,
    ) -> list:
        """
        A feature S (not in PC) is a spouse of T if there exists
        some Z in PC such that S and T are dependent given Z.

        Intuition: spouses are features that become relevant only
        in the context of a parent/child — e.g. in a V-structure X→Z←S,
        S is irrelevant to T marginally but relevant given Z.
        """
        spouses = []
        for f in candidates:
            if f in PC:
                continue
            for z in PC:
                score = self._cmi(
                    X_local[:, f], y,
                    X_local[:, [z]]
                )
                if score > self.alpha:
                    spouses.append(f)
                    print(f"  [Spouse] added feature {f} "
                          f"via conditioning on {z}  CMI={score:.4f}")
                    break   # one witness is enough
        return spouses

    # ---------------------------------------------------------- main: HITON-MB

    def _hitonmb(
        self,
        cluster_indices: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> list:
        """
        Runs HITON-MB on a single cluster.
        Returns selected global feature indices.
        """
        if len(cluster_indices) == 1:
            print(f"  Single feature {cluster_indices[0]} — kept directly")
            return cluster_indices

        print(f"  Running on {len(cluster_indices)} features: {cluster_indices}")
        X_local    = X_train[:, cluster_indices]
        candidates = list(range(X_local.shape[1]))

        # Phase 1 — find PC
        PC = self._find_PC(X_local, y_train, candidates)
        print(f"  PC set (local indices): {PC}")

        # Phase 2 — find Spouses
        spouses = self._find_spouses(X_local, y_train, PC, candidates)
        print(f"  Spouse set (local indices): {spouses}")

        MB = sorted(set(PC + spouses))

        # fallback — if MB is empty keep highest MI feature
        if not MB:
            scores = [self._mi(X_local[:, f], y_train) for f in candidates]
            MB     = [int(np.argmax(scores))]
            print(f"  Fallback — kept highest MI feature "
                  f"(local {MB[0]} → global {cluster_indices[MB[0]]})")

        selected_global = [cluster_indices[i] for i in MB]
        return selected_global

    # ------------------------------------------------------------ public entry

    def run(
        self,
        clusters: dict,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list:
        """
        Runs HITON-MB on each cluster independently.
        Noise features (passed separately via CluMarPareto) bypass this step.

        Parameters
        ──────────
        clusters : {cluster_id: [global_feature_indices]}
        X        : (n_samples, n_features) training data
        y        : (n_samples,) target labels

        Returns
        ───────
        List of selected global feature indices (from clusters only).
        Noise features are added by CluMarPareto after this call.
        """
        # auto-derive hyperparameters from data
        self.alpha  = 0.05 / X.shape[1]
        self.n_bins = max(5, min(20, round(X.shape[0] ** (1 / 3))))

        print(f"[HITON-MB] auto alpha  = {self.alpha:.6f}  "
              f"(Bonferroni: 0.05 / {X.shape[1]} features)")
        print(f"[HITON-MB] auto n_bins = {self.n_bins}  "
              f"(cube-root rule: {X.shape[0]} samples)")

        selected = []
        for cid, indices in clusters.items():
            print(f"\n[HITON-MB] Cluster {cid} — {len(indices)} features")
            kept = self._hitonmb(indices, X, y)
            print(f"  → Selected {len(kept)}: {kept}")
            selected.extend(kept)

        return selected


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

def main():
    X, y = make_classification(
        n_samples=1000, n_features=30,
        n_informative=10, n_redundant=10,
        n_repeated=0, n_classes=2,
        random_state=42
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
        print(f"  Cluster {cid}: {[feature_cols[i] for i in idxs]}")
    if model.noise:
        print(f"  Noise: {[feature_cols[i] for i in model.noise]}")

    hiton = IAMB()
    selected = hiton.run(clusters=model.clusters, X=X_train, y=y_train)
    selected += model.noise
    selected  = sorted(set(selected))

    print(f"\nSelected ({len(selected)}): {[feature_cols[i] for i in selected]}")


if __name__ == "__main__":
    main()