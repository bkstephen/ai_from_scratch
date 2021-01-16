using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ai_from_scratch
{
    public class LayerDense
    {
        public LayerDense(IList<double[]> inputs, int neuronsCount)
        {
            for (int i = 0; i < neuronsCount; i++)
            {
                // Create random weights for each neuron
                var tempWeights = Vector.Random(inputs.Count());
                var newNeuron = new Neuron
                {
                    Weights = tempWeights,
                    Bias = 0,
                };
                Neurons.Add(newNeuron);
            }

            Forward(inputs);
        }

        public IList<Neuron> Neurons { get; set; } = new List<Neuron>();

        public int Ordinal { get; set; }

        public double[][] DotProduct { get; set; }

        public void Forward(IList<double[]> inputs)
        {
            var weights = Neurons.Select(s => s.Weights).ToArray();
            var biases = Neurons.Select(s => s.Bias).ToArray();
            DotProduct = Matrix.Dot(inputs.ToArray(), weights.Transpose())
                                .Select(val => val.Select((s, i) => s + biases[i]).ToArray())
                                .ToArray();
        }
    }
}
