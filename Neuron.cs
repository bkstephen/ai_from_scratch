using Accord.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ai_from_scratch
{
    public class Neuron
    {
        public IList<double> Weights { get; set; }

        public double Bias { get; set; }

        public double[] DotProduct { get; set; }
    }
}
