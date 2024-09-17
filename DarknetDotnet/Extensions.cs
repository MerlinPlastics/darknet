using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DarknetDotnet
{
	internal static class RectangleFExtensions
	{
		public static float Area(this RectangleF rect)
		{
			return rect.Width * rect.Height;
		}

		public static float IOU(this RectangleF value, RectangleF other)
		{
			float intersection = RectangleF.Intersect(value, other).Area();
			//float union = RectangleF.Union(value, other).Area() - intersection;
			float union = value.Area() + other.Area() - intersection;


			if (union <= 0)
			{
				return 0;
			}

			return intersection / union;
		}
	}
}
