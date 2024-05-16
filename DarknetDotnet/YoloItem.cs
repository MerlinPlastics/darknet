using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetDotnet
{
	public record YoloItem : IComparable<YoloItem>
	{
		// Use backing fields here, and optimize the Rectange & Point by clearing their cached values on setting
		// of any of these properties
		public int X { get; set; }

		public int Y { get; set; }

		public int Width { get; set; }

		public int Height { get; set; }

		public int ObjectTypeId { get; set; }

		public string? Type { get; set; }

		public double Confidence { get; set; }

		private RectangleF? _rectangle;
		public RectangleF Rectangle
		{
			get
			{
				// Perf optimization
				return (RectangleF)(_rectangle = _rectangle ?? new RectangleF(X, Y, Width, Height));
				//return new RectangleF(X, Y, Width, Height);
			}
		}

		private Point? _point;
		public Point Center
		{
			get
			{
				return (Point)(_point = _point ?? new Point(this.X + this.Width / 2, this.Y + this.Height / 2));
				//return new Point(this.X + this.Width / 2, this.Y + this.Height / 2);
			}
		}

		public float IOU(YoloItem other)
		{
			var intersection = RectangleF.Intersect(this.Rectangle, other.Rectangle).Area();
			var union = RectangleF.Union(this.Rectangle, other.Rectangle).Area();

			if (union == 0)
				return -1;

			return intersection / union;
		}

		public float DistanceTo(YoloItem? other)
		{
			if (other == null) return float.MaxValue;

			return (float)Math.Sqrt(Math.Pow(other.X - this.X, 2) + Math.Pow(other.Y - this.Y, 2));
		}

		//public object Clone()
		//{
		//	return new YoloItem()
		//	{
		//		Type = this.Type,
		//		Confidence = this.Confidence,
		//		Height = this.Height,
		//		Width = this.Width,
		//		X = this.X,
		//		Y = this.Y
		//	};
		//}

		public int CompareTo(YoloItem? other)
		{
			if (other == null) return 1;

			if (other.Confidence == this.Confidence) return 0;

			// If other's confidence is higher, it comes first (-1)
			return (other.Confidence > this.Confidence) ? -1 : 1;

		}

		public static bool operator >(YoloItem operand1, YoloItem operand2)
		{
			return operand1.CompareTo(operand2) > 0;
		}

		public static bool operator <(YoloItem operand1, YoloItem operand2)
		{
			return operand1.CompareTo(operand2) < 0;
		}

		public static bool operator >=(YoloItem operand1, YoloItem operand2)
		{
			return operand1.CompareTo(operand2) >= 0;
		}

		public static bool operator <=(YoloItem operand1, YoloItem operand2)
		{
			return operand1.CompareTo(operand2) <= 0;
		}
	}
}
