using System;

namespace EasyNN {
    public interface NeuralNetwork {
        NeuralNetworkSettings.NeuralNetworkTypes NeuralNetworkType();
        NeuralNetworkBase[] NeuralNetworkBases();
        NeuralNetworkSettings.OptimizationMethods OptimizationMethod();
        NeuralNetworkSettings.CostFunctions CostFunction();
        NeuralNetworkSettings.ActivationFunctions ActivationFunction_Input();
        NeuralNetworkSettings.ActivationFunctions ActivationFunction_Hidden();
        NeuralNetworkSettings.ActivationFunctions ActivationFunction_Output();
        float LearningRate();

        float[] ForwardPropagate(float[] Inputs);
        void Train(int amount);
    }
}