# masters-research-project-machine-learning-for-process-simulation-and-optimization

This project was split into three parts of investigation. The results are as follows:

Part 1:
Accuracy and speed of surrogate modelling for three different ML algorithms (random forests, Gaussian processes and neural networks) and 3 different sampling techniques (random, stratified, latin hypercube).
In this part, the physical system being surrogate modelled was an ethylene oxide plug flow reactor.

Part 2: Use of a random forest surrogate model to provide initial guesses for numerical solution, in order to improve the success rate for convergence across the input parameter space.
In this part, the physical system being surrogate modelled was an ethylene oxide flash drum.

Part 3: Use of random forest regression to surrogate model the objective function resulting from a non-linear optimization. Use of random forest classification to predict the convergence status of an optimization, thus saving time by avoiding infeasible initial conditions.
In this part, the physical system being surrogate modelled was the purification section of an ethylene oxide process plant, consisting of a distillation column, absorption column, valve, heat exchanger and recycle stream.
