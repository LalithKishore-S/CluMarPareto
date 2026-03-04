# CluMarPareto
Cluster-Guided Markov Blanket Assisted Multi-Objective Feature Selection Using NSGA-II

`NSGA2.py`
- Progress:
    - 2 objective optimization 
        - Max CV accuracy
        - Min num_features selected
    - Should extend to 3 or more objectives
    - Added code for non dominated sorting 
    - Added code for crowding distance computation 
    - Completed NSGA2 pipeline after crowding distance
- To do:
    - Add code for Decision Trees or other classifiers
    - Explore NSGA3 or other improvements over crowding distance so that it NSGA works better for more than 2 objectives
