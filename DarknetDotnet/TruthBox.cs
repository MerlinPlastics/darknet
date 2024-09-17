using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetDotnet
{
	class TruthBox
	{
		public int IndexId { get; set; }
		public int ClassId { get; set; }
		public float X { get; set; }
		public float Y { get; set; }
		public float Width { get; set; }
		public float Height { get; set; }

		public RectangleF Rectangle
		{
			get
			{
				return new RectangleF(X, Y, Width, Height);
			}
		}

		public static TruthBox? Parse(string line, int? indexId = -1)
		{
			if (string.IsNullOrWhiteSpace(line))
				return null;

			var parts = line.Split(' ');

			if (parts.Length != 5)
				throw new ArgumentException("Line should have 5 parts");

			return new TruthBox()
			{
				IndexId = indexId ?? -1,
				ClassId = int.Parse(parts[0]),
				X = float.Parse(parts[1]),
				Y = float.Parse(parts[2]),
				Width = float.Parse(parts[3]),
				Height = float.Parse(parts[4]),
			};
		}
	}
}
