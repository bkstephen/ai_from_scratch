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
                var tempWeights = new List<double>();
                for (int weightsCount = 0; weightsCount < inputs.Count(); weightsCount++)
                {
                    tempWeights.Add(new Random().NextDouble() * (new Random().Next(-1, 1) < 0 ? -1 : 1));
                }
                var newNeuron = new Neuron
                {
                    Weights = tempWeights,
                    Bias = 0,
                };
                newNeuron.DotProduct = Matrix.Dot(inputs.ToArray(), newNeuron.Weights.ToArray())
                                        .Select(s => s + newNeuron.Bias)
                                        .ToArray();
                if (Neurons == null)
                {
                    Neurons = new List<Neuron>();
                }

                Neurons.Add(newNeuron);
            }
        }

        public IList<Neuron> Neurons { get; set; }

        public int Ordinal { get; set; }
    }
}
