# Neural-Network-Hyper-Parameter-Optimization
In this second part of a bigger project, a general neural network with back propagation was implemented from scratch in python.  Two methods of training, one with just the learning rate alpha and one with a momentum parameter mu for momentum based training were defined for the neural network. Then, a tuning algorithm that relied on searching the hyper parameter space using binary search ideas was innovated, and the performance was compared with a traditional grid search algorithm using an abstract strategy game.

### The Neural Network
The class neural_network defines general neural network objects with can be initialized to match any problem context. Each neural network object has a forward proagation method and a gradient cost calculation method (used when back propagating). In this case, the sigmoid cost function was used, although present literature now calls for the use of ReLU cost functions (easily changable). Upon initialization, a nerual network object will take in the number of input layers, an array of hidden layers H with H[i] representing the number of hidden nodes in layer i, and the number of output nodes. It will then be initialized with random and normalized weights between connecting nodes. The functions no_momentum_train and momentum_train take in a neural_network object, Input and Output data, and appropriate hyper parameters (either only alpha or both alpha and mu), and a number of iterations, and train the neural network with just the the learning rate or both the learning rate and momentum, respectively, and will return the resulting trained network.

This framework is very general and can be used with any sort of problem or data. In this case it was used to predict the optimal next moves of a player in an abstract strategy game (like chess).

### Hyper Parameter Optimization
This module defines hyper parameter tuning functions which can be used to tune a neural network. Namely, there is an implementation of a traditional grid search algorithm grid_search, which takes in an evenly partitioned set of the hyper parameter sample space and initializes a neural network with a hyper parameter value from the partition. Each of these neural networks is then evaluated with the function network_evaluation which evaluates the network on the basis of both speed and accuracy (on given input and output data). The grid_search function then returns the hyper parameter which resulted in the network with the best performance.

Then there is the innovated algorithm bin_search. This algorithm relies on the assumption that the hyper parameter sample space that it is searching in is practically convex and smooth, and uses a methodology similar to (albeit not the same as) a binary search to search the space exponentially faster than the grid_search algorithm. However note that the function that is being optimized here (the one that the binary search methodology is operating on) is f(x) where f is network_evaluate function and x is a value from

### Results

<!-- Improvement using gradient descent with network_evaluate, perhaps heuristically combine methods -->
<!-- Incomplete -->
