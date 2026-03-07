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
    - Knee point based single optimal solution finding.
- To do:
    - Explore NSGA3 or other improvements over crowding distance so that it NSGA works better for more than 2 obj

- Observations:
    - With two objectives (i.e CV accuracy and num_features), NSGA2 selects higher number of random features also in addition to informative features => fails to learn noise signals from informative ones with fewer number of generations.
    - So should try NSGA3, Enhanced NSGA2, CNSGA2 and A 2025 ScienceDirect paper proposes combining filter methods (Information Gain, Random Forest importance, Relief-F) with NSGA-II ScienceDirect — using filter scores to initialize the population intelligently rather than randomly.


