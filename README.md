# Implementation-of-the-Wang-Landau-algorithm
The Wang-Landau algorithm based on the Monte Carlo method is proposed by Fugao Wang  and David Landau [1] to calculate density of the states(DOS) efficiently. This method constructs  the DOS through non-Markov random transitions, traversing all possible states. After reviewing  Wang-Landau algorithm, the analysis of the results through the implementation in Python for 2D  Ising model is shown in this short report.

Environment: Python 3.6; numpy >= 1.19.5; matplotlib >= 3.3.4; 

Default Parameters: 
MCsweeps = 1000000   # Total MC sweeps
L = 256              # 2D Ising model lattice = L x L
flatness = 0.8      # “flat histogram”: histogram H(E) for all possible E is not less than 80% of the average histogram
N = L * L           # 2D Ising model lattice = L x L
Can change the number to test as your wish.

References
[1] F.Wang, and D.P.Landau, Efficient, Multiple-Range Random Walk Algorithm to Calculate the 
Density of States, Physical Review Letters 86, 2050 (2001).
