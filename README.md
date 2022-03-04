# EasyNN
## How to use
### Creating a neural network
To create a basic feedforward neural network, create a class with the interface NeuralNetwork.
Thats it! You just need to specify the desired size of the network, the cost function, and the 
activation functions.
### Training a neural network
Put your training data in the TrainingData class and call NeuralNetworkMath.Propagation.BackPropagation() 
in the Train() function in your neural network class using the TrainingData class as a paramater.
### Feedforward
Call NeuralNetworkMath.Propagation.ForwardPropagation() in the ForwardPropagate() function in your
neural network class using your desired inputs as paramters.
