import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class NSGA2_FS():
    def __init__(self, classifier = 'randomforest', population_size = 1000):
        self.n_obj = 2
        self.classifier = classifier
        self.N = population_size

    def generate_populations(self):
        population = []
        for _ in range(self.N):
            while True:
                individual = np.random.randint(0, 2, size = self.n_cols).astype(bool)
                if individual.sum() >= 1:
                    break
            population.append(individual)
        return np.array(population)
    
    def fitness_evaluation(self, individual, X, y):
        n_features_selected = int(individual.sum())
        X_masked = X[:, individual]
        model = RandomForestClassifier(n_estimators = 100, bootstrap = True, oob_score = True, n_jobs = -1, random_state = 42)
        model.fit(X_masked, y)

        oob_score = model.oob_score_

        return (n_features_selected, oob_score)
    
    def dominates(self, a, b):
        """
        A dominates B if:
          - A is no worse than B on ALL objectives, AND
          - A is strictly better on AT LEAST ONE objective
        """
        a_features,  a_acc = a
        b_features,  b_acc = b
        a_obj = np.array([ a_features, -a_acc])
        b_obj = np.array([ b_features, -b_acc])

        no_worse    = np.all(a_obj <= b_obj)
        strictly_better = np.any(a_obj <  b_obj)

        return no_worse and strictly_better
    
    
    def non_dominated_sorting(self, objective_scores):

        domination_count = np.zeros(self.N, dtype=int)   # how many individuals dominate p
        dominated_set    = [[] for _ in range(self.N)]   # individuals that p dominates

        fronts = [[]]

        for p in range(self.N):
            for q in range(self.N):
                if p == q:
                    continue
                if self.dominates(objective_scores[p], objective_scores[q]):
                    dominated_set[p].append(q)
                elif self.dominates(objective_scores[q], objective_scores[p]):
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)                 

        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for p in fronts[current_front]:
                for q in dominated_set[p]:
                    domination_count[q] -= 1        
                    if domination_count[q] == 0:    
                        next_front.append(q)
            current_front += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]
        
    def fit(self, data):
        self.columns = list(data.columns)
        self.n_cols = len(self.columns) - 1
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values

        initial_population = self.generate_populations()
        objective_scores = [self.fitness_evaluation(individual, X, y) for individual in initial_population]
        fronts = self.non_dominated_sorting(objective_scores)

        print(f"\nClassifier      : {self.classifier}")
        print(f"Population size : {self.N}")
        print(f"Number of fronts: {len(fronts)}\n")

        for rank, front in enumerate(fronts):
            print(f"  Rank {rank+1} -- {len(front)} individual(s):")
            for idx in front:
                n_feat, acc = objective_scores[idx]
                print(f"    individual {idx:>3} | features: {n_feat:>2} | accuracy: {acc:.4f}")

        


def main():
    def generate_dataset():
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,   
            n_redundant=5,      
            n_repeated=0,
            n_classes=2,
            random_state=42
        )

        feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df["target"] = y
        return df

    df = generate_dataset()

    print("Shape     :", df.shape)
    print("Target dist:\n", df["target"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())

    nsga = NSGA2_FS(classifier = 'decisiontree', population_size = 200)
    nsga.fit(df)


main()