using System.Linq;
using System;
using System.Collections.Generic;
using Accord.Math;

namespace ai_from_scratch
{
    class Program
    {
        private static double[][] inputs = new[]
        {
            new[] {1.0, 2.0, 3.0, 2.5},
            new[] {2.0, 5.0, -1.0, 2.0},
            new[] {-1.5, 2.7, 3.3, -0.8}
        };
        // Each neuron has its own weight and bias
        private static double[][] weights = new[]
        {
            new[] { 0.2, 0.8, -0.5, 1.0 },
            new[] { 0.5, -0.91, 0.26, -0.5 },
            new[] { -0.26, -0.27, 0.17, 0.87 }
        };
        private static double[] biases = new[] { 2.0, 3.0, 0.5 };

        private static double[][] weights2 = new[]
        {
            new[] { 0.1, -0.14, 0.5},
            new[] { -0.5, 0.12, -0.33},
            new[] { -0.44, 0.73, -0.13}
        };
        private static double[] biases2 = new[] { -1, 2, -0.5 };

        static void Main(string[] args)
        {
            var newLayer = new LayerDense(inputs, 6);
            var result = newLayer.DotProduct;

            //var weighted_output = Matrix.Dot( inputs, weights.Transpose())
            //        .Select(val=> val.Select((s, i) => s + biases[i]).ToArray())
            //        .ToArray();            

            foreach (var items in result)
            {
                foreach (var item in items)
                {
                    Console.Write($"{item.ToString()}, ");
                }
                Console.WriteLine();
            }
        }
    }
}
