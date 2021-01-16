using System.Linq;
using System;
using System.Collections.Generic;
using Accord.Math;

namespace ai_from_scratch
{
    class Program
    {
        private static double[] inputs = new[] {1.0, 2.0, 3.0, 2.5};

        // Each neuron has its own weight and bias
        private static double[][] weights = new[] {
            new double[] { 0.2, 0.8, -0.5, 1 },
            new double[] { .5, -0.91, 0.26, -0.5 },
            new double[] { -0.26, -0.27, 0.17, 0.87 }
        };
        private static double[] biases = new[] {2.0, 3.0, 0.5};

        static void Main(string[] args)
        {
            List<double> layerOutputs = new List<double>();
            var weighted_output = Matrix.Dot(weights, inputs);
            foreach (var (neuronOutput, bias) in weighted_output.Zip(biases))
            {
                layerOutputs.Add(neuronOutput + bias);
            }
            
            // foreach (var (neuronWeight, neuronBias) in weights.Zip<double[], double>(biases))
            // {
            //     var neuronOutput = 0.0;
            //     foreach (var (nInput, weight) in inputs.Zip<double, double>(neuronWeight)){
            //         neuronOutput += nInput*weight;
            //     }
            //     
            //     neuronOutput += neuronBias;
            //     layerOutputs.Add(neuronOutput);
            // }

            foreach (var item in layerOutputs)
            {
                Console.Write($"{item.ToString()}, ");    
            }

            Console.Read();
        }
    }
}
