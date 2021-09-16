using System;

namespace EasyNN {
    public class NeuralNetworkMath {
        public class MatrixMath
        {
            //Returns the index of the weight between to neurons
            public static int GetWeightBetweenTwoNeurons(NeuralNetworkBase nn, int Layer, int Neuron0, int Neuron1)
            {
                return (Neuron0 * nn.Neurons[Layer+1].Length + Neuron1);
            }
            //Returns the neuron from the previous layer from a given weight
            public static int GetIncomingNeuronFromWeight(NeuralNetworkBase nn, int Layer, int Weight)
            {
                return (int)Math.Floor((float)(Weight / nn.Neurons[Layer+1].Length));
            }
            //Returns the neuron from the next layer from a given weight
            public static int GetOutgoingNeuronFromWeight(NeuralNetworkBase nn, int Layer, int Weight)
            {
                return (Weight % nn.Neurons[Layer+1].Length);
            }
        }
        public class BackPropMath
        {
            //This function returns the change in the cost function with respect to the change in a weight
            public static float dCx_dw(NeuralNetworkBase nn, NeuralNetwork nn_interface, int Layer, int Weight, float[][] dcxda_matrix)
            {
                return (
                    dz_dw(nn, Layer, Weight) *
                    da_dz(nn, nn_interface, Layer+1, MatrixMath.GetOutgoingNeuronFromWeight(nn, Layer, Weight)) *
                    dcxda_matrix[Layer+1][MatrixMath.GetOutgoingNeuronFromWeight(nn, Layer, Weight)]
                    );
            }
            //This function return the change in the cost function with respect to the change in the bias
            public static float dCx_db(NeuralNetworkBase nn, NeuralNetwork nn_interface, int Layer, int Bias, float[][] dcxda_matrix)
            {
                return (
                    dz_db(nn, Layer, Bias) *
                    da_dz(nn, nn_interface, Layer, Bias) *
                    dcxda_matrix[Layer][Bias]
                    );
            }
            //This function returns the change in the cost function with respect to the change in an activation
            public static float dCx_da(NeuralNetworkBase nn, NeuralNetwork nn_interface, int Layer, int Neuron, float[][] dcxda_matrix)
            {
                float T = 0;
                
                for (int i = 0; i < nn.Neurons[Layer+1].Length; i++)
                {
                    T += (
                        dz_da(nn, Layer+1, i, Neuron) *
                        da_dz(nn, nn_interface, Layer+1, i) *
                        dcxda_matrix[Layer+1][i]
                        );
                }
                return T;
            }
            //This function returns the change in the cost function with respect to the change in an activation
            public static float dCx_da(NeuralNetworkBase nn, NeuralNetworkSettings.CostFunctions costFunction, TrainingEpoch epoch, int Neuron)
            {
                //Console.WriteLine(Functions.CostFunctionPrime(costFunction, nn.Neurons[nn.Neurons.Length - 1][Neuron], epoch.ExpectedValue[Neuron]));
                return Functions.CostFunctionPrime(costFunction, nn.Neurons[nn.Neurons.Length - 1][Neuron], epoch.ExpectedValue[Neuron]);
            }
            //This function represents the change in z with respect to the change in a weight
            public static float dz_dw(NeuralNetworkBase nn, int Layer, int Weight)
            {
                return nn.Neurons[Layer][MatrixMath.GetIncomingNeuronFromWeight(nn, Layer, Weight)];
            }
            //This function represents the change in z with respect to the change in bias
            public static float dz_db(NeuralNetworkBase nn, int Layer, int Bias)
            {
                return nn.Bias[Layer][Bias];
            }
            //This function represents the change in z with respect to the change in the activation
            public static float dz_da(NeuralNetworkBase nn, int Layer, int Neuron, int IncomingNeuron)
            {
                return nn.Weights[Layer - 1][MatrixMath.GetWeightBetweenTwoNeurons(nn, Layer - 1, IncomingNeuron, Neuron)];
            }
            //This function returns the change in the activation with respect to the change in z.
            public static float da_dz(NeuralNetworkBase nn, NeuralNetwork nn_interface, int Layer, int Neuron)
            {
                if (Layer == 0)
                {
                    return Functions.ActivationFunctionPrime(nn_interface.ActivationFunction_Input(), ForwardPropMath.Z(nn, nn_interface, Layer, Neuron));
                }
                else if (Layer == nn.Neurons.Length - 1)
                {
                    return Functions.ActivationFunctionPrime(nn_interface.ActivationFunction_Output(), ForwardPropMath.Z(nn, nn_interface, Layer, Neuron));
                }
                else
                {
                    return Functions.ActivationFunctionPrime(nn_interface.ActivationFunction_Hidden(), ForwardPropMath.Z(nn, nn_interface, Layer, Neuron));
                }
            }
        }
        public class ForwardPropMath
        {
            //This function returns the Z at the specified layer and neuron.
            public static float Z(NeuralNetworkBase nn, NeuralNetwork nn_interface, int Layer, int Neuron)
            {
                if (Layer == 0)
                {
                    return nn.Neurons[Layer][Neuron];
                }
                else
                {
                    float T = 0;

                    for (int neuron = 0; neuron < nn.Neurons[Layer - 1].Length; neuron++)
                    {
                        T += (
                            nn.Neurons[Layer - 1][neuron]
                            *
                            nn.Weights[Layer - 1][MatrixMath.GetWeightBetweenTwoNeurons(nn, Layer - 1, neuron, Neuron)]
                            );
                    }

                    T += nn.Bias[Layer][Neuron];

                    return T;
                }
            }
        }
        public class Functions
        {
            public class ActivationFunctions
            {
                //This function returns the value of the sigmoid activation function using the specified values
                public static float Sigmoid(float z)
                {
                    float Numerator = 1;
                    float Denominator = (1 + (float)Math.Pow((float)Math.E, -1 * z));
                    float returnValue = Numerator / Denominator;
                    return returnValue;
                }
                //This function returns the value of the relu activation function using the specified values
                public static float ReLU(float z)
                {
                    if (z <= 0)
                    {
                        return 0;
                    }
                    else
                    {
                        return z;
                    }
                }
                //This function returns the value of the leakyrelu activation function using the specified values
                public static float LeakyReLU(float z)
                {
                    if (z <= 0)
                    {
                        return 0.01f*z;
                    }
                    else
                    {
                        return z;
                    }
                }
                //This function returns the value of the tanh activation function using the specified values
                public static float tanh(float z)
                {
                    float Numerator = ((float)Math.Pow(Math.E, z) - (float)Math.Pow(Math.E, -1 * z));
                    float Denominator = ((float)Math.Pow(Math.E, z) + (float)Math.Pow(Math.E, -1 * z));
                    float returnValue = Numerator / Denominator;
                    return returnValue;
                }
                public class Prime
                {
                    //This function returns the derivative of the sigmoid activation function using the specified values
                    public static float Sigmoid(float z)
                    {
                        return (ActivationFunctions.Sigmoid(z) * (1 - ActivationFunctions.Sigmoid(z)));
                    }
                    //This function returns the derivative of the relu activation function using the specified values
                    public static float ReLU(float z)
                    {
                        if (z <= 0)
                        {
                            return 0;
                        }
                        else
                        {
                            return 1;
                        }
                    }
                    //This function returns the derivative of the leakyrelu activation function using the specified values
                    public static float LeakyReLU(float z)
                    {
                        if (z <= 0)
                        {
                            return 0.01f;
                        }
                        else
                        {
                            return 1;
                        }
                    }
                    //This function returns the derivative of the tanh activation function using the specified values
                    public static float tanh(float z)
                    {
                        return (1 - ActivationFunctions.tanh(z)*ActivationFunctions.tanh(z));
                    }
                }
            }
            public class CostFunctions
            {
                //This function returns the value of the quadratic loss function at the specified values
                public static float QuadraticLoss(float val, float ex_val)
                {
                    return (float)Math.Pow((ex_val - val), 2);
                }
                public class Prime
                {
                    //This function returns the derivative of the quadratic loss cost function using the specified values
                    public static float QuadraticLoss(float val, float ex_val)
                    {
                        return 2 * (ex_val - val);
                    }
                }
            }
            //This function returns the desired Activation function
            public static float ActivationFunction(NeuralNetworkSettings.ActivationFunctions activation, float z)
            {
                switch(activation)
                {
                    case NeuralNetworkSettings.ActivationFunctions.Sigmoid:
                        return ActivationFunctions.Sigmoid(z);
                    case NeuralNetworkSettings.ActivationFunctions.LeakyReLU:
                        return ActivationFunctions.LeakyReLU(z);
                    case NeuralNetworkSettings.ActivationFunctions.ReLU:
                        return ActivationFunctions.ReLU(z);
                    case NeuralNetworkSettings.ActivationFunctions.tanh:
                        return ActivationFunctions.tanh(z);
                    default:
                        return ActivationFunctions.Sigmoid(z);
                }
            }
            //This function returns the derivative of the desired activation function
            public static float ActivationFunctionPrime(NeuralNetworkSettings.ActivationFunctions activation, float z)
            {
                switch (activation)
                {
                    case NeuralNetworkSettings.ActivationFunctions.Sigmoid:
                        return ActivationFunctions.Prime.Sigmoid(z);
                    case NeuralNetworkSettings.ActivationFunctions.LeakyReLU:
                        return ActivationFunctions.Prime.LeakyReLU(z);
                    case NeuralNetworkSettings.ActivationFunctions.ReLU:
                        return ActivationFunctions.Prime.ReLU(z);
                    case NeuralNetworkSettings.ActivationFunctions.tanh:
                        return ActivationFunctions.Prime.tanh(z);
                    default:
                        return ActivationFunctions.Prime.Sigmoid(z);
                }
            }
            //This function returns the value of the cost function
            public static float CostFunction(NeuralNetworkSettings.CostFunctions cost, float val, float ex_val)
            {
                switch (cost)
                {
                    case NeuralNetworkSettings.CostFunctions.QuadraticLoss:
                        return CostFunctions.QuadraticLoss(val, ex_val);
                    default:
                        return CostFunctions.QuadraticLoss(val, ex_val);
                }
            }
            //This function is for testing purposes. It returns the average of the output layer
            public static float CostFunctionTotal(NeuralNetworkBase nn, NeuralNetworkSettings.CostFunctions cost, TrainingEpoch epoch)
            {
                float T = 0;
                for (int i = 0; i < nn.Neurons[nn.Neurons.Length - 1].Length; i++)
                {
                    switch (cost)
                    {
                        case NeuralNetworkSettings.CostFunctions.QuadraticLoss:
                            T += CostFunctions.QuadraticLoss(nn.Neurons[nn.Neurons.Length - 1][i], epoch.ExpectedValue[i]);
                            break;
                        default:
                            T += CostFunctions.QuadraticLoss(nn.Neurons[nn.Neurons.Length - 1][i], epoch.ExpectedValue[i]);
                            break;
                    }
                }
                return T/nn.Neurons[nn.Neurons.Length - 1].Length;
            }
            //This function returns the derivative of the desired cost function
            public static float CostFunctionPrime(NeuralNetworkSettings.CostFunctions cost, float val, float ex_val)
            {
                switch (cost)
                {
                    case NeuralNetworkSettings.CostFunctions.QuadraticLoss:
                        return CostFunctions.Prime.QuadraticLoss(val, ex_val);
                    default:
                        return CostFunctions.Prime.QuadraticLoss(val, ex_val);
                }
            }
        }
        public class Propagation
        {
            //This function forward propagates NeuralNetworkBase nn using inputs
            public static NeuralNetworkBase ForwardPropagation(NeuralNetworkBase nn, NeuralNetwork nn_interface, float[] Inputs)
            {
                NeuralNetworkBase new_nn = nn;

                if (Inputs.Length != new_nn.Neurons[0].Length)
                {
                    return new_nn;
                }
                new_nn.Neurons[0] = Inputs;
                for (int Layer = 1; Layer < new_nn.Neurons.Length; Layer++)
                {
                    NeuralNetworkSettings.ActivationFunctions activation = NeuralNetworkSettings.ActivationFunctions.Sigmoid;
                    if (Layer == 0)
                    {
                        activation = nn_interface.ActivationFunction_Input();
                    }
                    else if (Layer == new_nn.Neurons.Length - 1)
                    {
                        activation = nn_interface.ActivationFunction_Output();
                    }
                    else
                    {
                        activation = nn_interface.ActivationFunction_Hidden();
                    }
                    for (int Neuron = 0; Neuron < new_nn.Neurons[Layer].Length; Neuron++)
                    {
                        new_nn.Neurons[Layer][Neuron] = (
                          NeuralNetworkMath.Functions.ActivationFunction(
                            activation,
                            NeuralNetworkMath.ForwardPropMath.Z(new_nn, nn_interface, Layer, Neuron)
                          )
                        );
                    }
                }

                return new_nn;
            }
            //This function backpropagates NeuralNetworkBase nn using epoch as the training data
            public static void BackPropagation(ref NeuralNetworkBase nn, NeuralNetwork nn_interface, TrainingEpoch epoch)
            {
                nn = ForwardPropagation(nn, nn_interface, epoch.Input);
                

                float[][] dCxda_matrix = new float[nn.Neurons.Length][];
                for (int i = 0; i < dCxda_matrix.Length; i++)
                {
                    dCxda_matrix[i] = new float[nn.Neurons[i].Length];
                }

                for (int Layer = dCxda_matrix.Length - 1; Layer > 0; Layer--)
                {
                    for (int Neuron = 0; Neuron < dCxda_matrix[Layer].Length; Neuron++)
                    {
                        float dcxda = 0;

                        if (Layer == dCxda_matrix.Length - 1)
                        {
                            dcxda = BackPropMath.dCx_da(nn, nn_interface.CostFunction(), epoch, Neuron);
                        }
                        else
                        {
                            for (int OutgoingNeuron = 0; OutgoingNeuron < dCxda_matrix[Layer + 1].Length; OutgoingNeuron++)
                            {
                                dcxda += BackPropMath.dCx_da(nn, nn_interface, Layer, Neuron, dCxda_matrix);
                            }
                        }

                        dCxda_matrix[Layer][Neuron] = dcxda;
                    }
                }

                for (int Layer = 0; Layer < nn.Weights.Length - 1; Layer++)
                {
                    for (int Weight = 0; Weight < nn.Weights[Layer].Length; Weight++)
                    {
                        nn.Weights[Layer][Weight] += BackPropMath.dCx_dw(nn, nn_interface, Layer, Weight, dCxda_matrix) * nn_interface.LearningRate();
                    }
                }

                for (int Layer = 1; Layer < nn.Bias.Length; Layer++)
                {
                    for (int Bias = 0; Bias < nn.Bias[Layer].Length; Bias++)
                    {
                        nn.Bias[Layer][Bias] += BackPropMath.dCx_db(nn, nn_interface, Layer, Bias, dCxda_matrix) * nn_interface.LearningRate();
                    }
                }
            }
        }
    }
}