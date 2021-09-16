using System;
using System.Collections.Generic;
using System.Text;

namespace EasyNN
{
    public class NeuralNetworkBase
    {
        public float[][] Neurons;
        public float[][] Weights;
        public float[][] Bias;

        public NeuralNetworkBase()
        {

        }
        public NeuralNetworkBase(int[] LayerSizes)
        {
            Neurons = new float[LayerSizes.Length][];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new float[LayerSizes[i]];

                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    Neurons[i][j] = 1;
                }
            }

            Weights = new float[LayerSizes.Length][];
            for (int i = 0; i < Neurons.Length - 1; i++)
            {
                Weights[i] = new float[LayerSizes[i] * LayerSizes[i + 1]];

                for (int j = 0; j < Weights[i].Length; j++)
                {
                    Random r = new Random();
                    float x = (float)r.Next(1, 9) * 0.1f;

                    Weights[i][j] = x;
                }
            }

            Bias = new float[LayerSizes.Length][];
            for (int i = 1; i < Neurons.Length; i++)
            {
                Bias[i] = new float[LayerSizes[i]];

                for (int j = 0; j < Bias[i].Length; j++)
                {
                    Random r = new Random();
                    float x = (float)r.Next(1, 9) * 0.1f;

                    Bias[i][j] = x;
                }
            }
        }
    }
}
