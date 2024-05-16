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
	}
}
