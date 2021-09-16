#Usage
Just use the interface NeuralNetwork to create your custom neural network class.
````c#
public NeuralNetwork CustomNN {
  public NeuralNetworkSettings.NeuralNetworkTypes NeuralNetworkType() {
    return NeuralNetowrkSettings.NeuralNetworkTypes.Feedforward;
  }
  NeuralNetworkBase[] NeuralNetworkBases() {
    int[] LayerSizes = new int[] { 3, 5, 1 };
    NeuralNetworkBase nnb = new NeuralNetworkBase(LayerSizes);
    NeturalNetworkBase[] rv  = new NeuralNetworkBase[] { nnb };
    return rv;
  }
  NeuralNetworkSettings.OptimizationMethods OptimizationMethod() {
    return NeuralNetworkSettings.OptimizationMethods.SthosticGradiantDescent;
  }
  NeuralNetworkSettings.CostFunctions CostFunction() {
    return NeuralNetworkSettings.CostFunctions.QuadraticLoss;
  }
  NeuralNetworkSettings.ActivationFunctions ActivationFunction_Input() {
    return NeuralNetworkSettings.ActivationFunctions.ReLU;
  }
  NeuralNetworkSettings.ActivationFunctions ActivationFunction_Hidden() {
    return NeuralNetworkSettings.ActivationFunctions.ReLU;
  }
  NeuralNetworkSettings.ActivationFunctions ActivationFunction_Output() {
    return NeuralNetworkSettings.ActivationFunctions.ReLU;
  }
}
````
