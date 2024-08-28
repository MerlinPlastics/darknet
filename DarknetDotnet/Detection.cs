using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetDotnet
{
	public class Detection
	{
		public float X { get; set; }
		public float Y { get; set; }
		public float W { get; set; }
		public float H { get; set; }

		public int ClassCount { get; set; }
		public float Objectness { get; set; }

		public Dictionary<int, float> Probabilities { get; set; } = new();
	}
}
