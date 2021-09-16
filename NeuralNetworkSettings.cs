using System;
using System.Collections.Generic;
using System.Text;

namespace EasyNN
{
    public class NeuralNetworkSettings
    {
        public enum NeuralNetworkTypes
        {
            Feedforward,
            Reccurant,
            Reccursive
        }
        public enum OptimizationMethods
        {
            SthosticGradiantDescent
        };
        public enum CostFunctions
        {
            QuadraticLoss
        };
        public enum ActivationFunctions
        {
            Sigmoid,
            ReLU,
            LeakyReLU,
            tanh
        };
    }
}
