# NeuraForge an easy neural network library

## NeuralNet Class
The NeuralNet class is a basic implementation of a feedforward neural network. It has methods to set the weights of the network, forward propagate an input, and print the weights and biases.

#### Initialization
To initialize a NeuralNet object, the following parameters can be passed:

- input (int): Number of input neurons in the network.
- output_layer (int): Number of neurons in the output layer of the network.
- hidden (list): List of integers representing the number of neurons in each hidden layer of the network. If not provided, the network will have no hidden layers.
- add_biases (bool): Whether to add biases to the network. If True, biases will be added to each neuron in the network.
- activation (function): The activation function to be used in the network. If not provided, the network will not apply any activation function.

#### Methods
`random(self, x, y=0)`
This method generates random values for the weights and biases of the network.

- x (int): Number of columns in the generated matrix.
- y (int, optional): Number of rows in the generated matrix. If not provided, a single row matrix is generated.
Returns a list of Value objects.

`setWeights(self)`
This method sets the weights and biases of the network based on the number of input and output neurons, as well as the number of neurons in any hidden layers.

`dotproduct(self, layer, weights)`
This method calculates the dot product of a layer with a weight matrix.

- layer (list): The layer to be multiplied.
- weights (list): The weight matrix to be multiplied.
Returns a list of dot products.

`forward(self, x)`
This method performs a forward pass through the network with the given input.

- x (list): The input to be propagated through the network.
Returns a list of Value objects representing the output of the network.

`printWeights(self)`
This method prints the weights of the network.

printBiases(self)
This method prints the biases of the network.

#### Attributes
- input (int): Number of input neurons in the network.
- output_layer (int): Number of neurons in the output layer of the network.
- hidden (list): List of integers representing the number of neurons in each hidden layer of the network.
- add_biases (bool): Whether to add biases to the network. If True, biases will be added to each neuron in the network.
- activation (function): The activation function to be used in the network. If not provided, the network will not apply any activation function.
- out (Value): A Value object representing the output of the network.

