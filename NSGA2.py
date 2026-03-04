import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

class Individual():
    def __init__(self, mask, obj_scores):
        self.mask_features = mask
        self.obj_scores = obj_scores
        self.crowding_distance = 0.0    
        self.rank = None

    def dominates(self, b):
        """
        A dominates B if:
          - A is no worse than B on ALL objectives, AND
          - A is strictly better on AT LEAST ONE objective
        """
        a_features,  a_acc = self.obj_scores
        b_features,  b_acc = b.obj_scores
        a_obj = np.array([ a_features, -a_acc])
        b_obj = np.array([ b_features, -b_acc])

        no_worse    = np.all(a_obj <= b_obj)
        strictly_better = np.any(a_obj <  b_obj)

        return no_worse and strictly_better

class NSGA2_FS():
    def __init__(self, classifier = 'randomforest', population_size = 1000):
        self.n_obj = 2
        self.classifier = classifier
        self.N = population_size
        self.n_generations = 5
        self.crossover_rate = 0.80
        self.mutation_rate = 0.5

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
    
    def tournament_select(self, population):
        i, j = np.random.choice(len(population), size=2, replace=False)
        a, b = population[i], population[j]
        if a.rank < b.rank:
            return a
        elif b.rank < a.rank:
            return b
        else:
            return a if a.crowding_distance >= b.crowding_distance else b

    def crossover(self, parent_a, parent_b):
        if np.random.rand() > self.crossover_rate:
            return parent_a.mask_features.copy(), parent_b.mask_features.copy()

        point   = np.random.randint(1, self.n_cols)
        child_a = np.concatenate([parent_a.mask_features[:point],
                                   parent_b.mask_features[point:]])
        child_b = np.concatenate([parent_b.mask_features[:point],
                                   parent_a.mask_features[point:]])
        return child_a, child_b

    def mutate(self, chromosome):
        mask  = np.random.rand(self.n_cols) < self.mutation_rate
        child = chromosome.copy()
        child[mask] = ~child[mask]
        if child.sum() == 0:                          
            child[np.random.randint(self.n_cols)] = True
        return child
      
    def create_offspring(self, population):
        children_masks = []
        while len(children_masks) < self.N:
            parent_a = self.tournament_select(population)
            parent_b = self.tournament_select(population)
            child_a, child_b = self.crossover(parent_a, parent_b)
            child_a = self.mutate(child_a)
            child_b = self.mutate(child_b)
            children_masks.append(child_a)
            if len(children_masks) < self.N:
                children_masks.append(child_b)
        return children_masks

    
    def select_next_generation(self, combined_population):
        fronts = self.non_dominated_sorting(combined_population)
        for front in fronts:
            self.crowded_distance_assignment(front, combined_population)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.N:
                next_gen.extend(front)
            else:
                remaining    = self.N - len(next_gen)
                sorted_front = sorted(front, key=lambda i: combined_population[i].crowding_distance, reverse=True)
                next_gen.extend(sorted_front[:remaining])
                break

        return [combined_population[i] for i in next_gen]

    def non_dominated_sorting(self, initial_population):
        n = len(initial_population)
        domination_count = np.zeros(n, dtype=int)    # how many individuals dominate p
        dominated_set    = [[] for _ in range(n)]    # individuals that p dominates

        fronts = [[]]

        for p in range(self.N):
            for q in range(self.N):
                if p == q:
                    continue
                if initial_population[p].dominates(initial_population[q]):
                    dominated_set[p].append(q)
                elif initial_population[q].dominates(initial_population[p]):
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

        for rank, front in enumerate(fronts):
            for idx in front:
                initial_population[idx].rank = rank
        return [f for f in fronts if len(f) > 0]

    def crowded_distance_assignment(self, front, individuals):
        n = len(front)
        distances = np.zeros(n, dtype=np.float32)

        for obj_idx in range(self.n_obj):
            sorted_order = sorted(range(n), key=lambda i: individuals[front[i]].obj_scores[obj_idx])

            distances[sorted_order[0]]  = np.inf
            distances[sorted_order[-1]] = np.inf

            obj_vals = [individuals[front[sorted_order[i]]].obj_scores[obj_idx] for i in range(n)]
            obj_range = obj_vals[-1] - obj_vals[0]

            if obj_range == 0:
                continue   

            for i in range(1, n - 1):
                distances[sorted_order[i]] += (obj_vals[i+1] - obj_vals[i-1]) / obj_range

        for i, idx in enumerate(front):
            individuals[idx].crowding_distance = distances[i]

              
    def fit(self, data):
        self.columns = list(data.columns)
        self.n_cols = len(self.columns) - 1
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values

        initial_population = self.generate_populations()
        objective_scores = [self.fitness_evaluation(individual, X, y) for individual in initial_population]
        initial_population = [Individual(initial_population[i], objective_scores[i]) for i in range(initial_population.shape[0])]
        fronts = self.non_dominated_sorting(initial_population)

        print(f"\nClassifier      : {self.classifier}")
        print(f"Population size : {self.N}")
        print(f"Number of fronts: {len(fronts)}\n")

        for rank, front in enumerate(fronts):
            print(f"  Rank {rank+1} -- {len(front)} individual(s):")
            for idx in front:
                n_feat, acc = objective_scores[idx]
                print(f"    individual {idx:>3} | features: {n_feat:>2} | accuracy: {acc:.4f}")
   
        for front in fronts:
            self.crowded_distance_assignment(front, initial_population)

        parent = initial_population

        for gen in range(self.n_generations):
            print(gen)
            print("-" * 50)
            children_masks  = self.create_offspring(parent)
            children_scores = [self.fitness_evaluation(m, X, y) for m in children_masks]
            children = [Individual(children_masks[i], children_scores[i]) for i in range(self.N)]

            combined = parent + children
            parent = self.select_next_generation(combined)

            if (gen + 1) % 10 == 0:
                pareto    = [ind for ind in parent if ind.rank == 0]
                best_acc  = max(ind.obj_scores[1] for ind in pareto)
                min_feats = min(ind.obj_scores[0] for ind in pareto)
                print(f"  Gen {gen+1:>3} | Pareto size: {len(pareto):>3} | "
                      f"Best acc: {best_acc:.4f} | Min features: {min_feats}")
        
        fronts = self.non_dominated_sorting(parent)
        pareto_front = [parent[i] for i in fronts[0]]
        # pareto_front.sort(key=lambda ind: ind.obj_scores[0])


        print(f"\nFinal Pareto Front ({len(pareto_front)} solutions):")
        print(f" {'Features_selected':>10} {'Features':>10} {'Accuracy':>10}")
        print("  " + "-" * 22)
        for ind in pareto_front:
            print(f" {ind.mask_features} {ind.obj_scores[0]:>10} {ind.obj_scores[1]:>10.4f}")

        self.pareto_front_ = pareto_front

def main():
    def generate_dataset():
        X, y = make_classification(
            n_samples=1000,
            n_features=30,
            n_informative=10,   
            n_redundant=5,      
            n_repeated=5,
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