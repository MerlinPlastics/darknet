using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using static DarknetDotnet.InteropMethods;

namespace DarknetDotnet
{
	internal class Program
	{
		public static void Main(string[] args)
		{
			string config = "LegoGears.cfg";
			string weights = "LegoGears_best.weights";
			int gpu = 0;

			string imageFile = @"C:\dev\darknet-merlin\DarknetDotnet\bin\Debug\net8.0\image.jpg";

			Program p = new Program();
			p.TestDetector(config, weights, gpu, imageFile);

		}

		internal void TestDetector(string config, string weights, int gpu, string imageFile)
		{
			IntPtr detector = IntPtr.Zero;
			try
			{
				detector = InteropMethods.CreateDetector(config, weights, gpu);
				var mat = Cv2.ImRead(imageFile, ImreadModes.Color);

				int count = 0;
				while (true)
				{

					InteropMethods.BboxContainer container2 = new InteropMethods.BboxContainer();
					int resultsB = InteropMethods.DetectorFromMat(detector, mat.CvPtr, 0.0f, ref container2);

					if (count++ % 1000 == 0)
					{
						Console.Write(".");
						GC.Collect(2);

						//InteropMethods.DisposeDetector(detector);
						//detector = InteropMethods.CreateDetector(config, weights, gpu);
					}
				}
			}
			catch (Exception ex)
			{
				InteropMethods.DisposeDetector(detector);
			}
		}

		//public static void MarshalUnmananagedArrayToStruct<T>(IntPtr unmanagedArray, int length, out T[] mangagedArray)
		//{
		//	var size = Marshal.SizeOf(typeof(T));
		//	mangagedArray = new T[length];

		//	for (int i = 0; i < length; i++)
		//	{
		//		IntPtr ins = new IntPtr(unmanagedArray.ToInt64() + i * size);
		//		mangagedArray[i] = Marshal.PtrToStructure<T>(ins);
		//	}

		//}


	}
}
