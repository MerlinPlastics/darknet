using System;
using System.Collections.Generic;
using System.Diagnostics;
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

			string imageFile = @"image.jpg";

			Program p = new Program();
			p.TestDetector(config, weights, gpu, imageFile);

		}

		internal void TestDetector(string config, string weights, int gpu, string imageFile)
		{
			var mat = Cv2.ImRead(imageFile, ImreadModes.Color);
			//while (true)
			//{
			using (var detector = YoloDetector.CreateDetector(".", 0))
			{
				int trials = 1000;
				Stopwatch sw = Stopwatch.StartNew();

				{
					sw.Restart();
					_ = detector.SpeedTest(trials);
					sw.Stop();

					var time = sw.Elapsed;

					Console.WriteLine("SpeedTest");
					Console.WriteLine($"{time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
				}

				{
					sw.Restart();
					for (int i = 0; i < trials; i++)
						_ = detector.Detect(mat, 0.2f);

					sw.Stop();
					var time = sw.Elapsed;

					Console.WriteLine("YoloDetector");
					Console.WriteLine($"{time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
				}

				//{
				//	sw.Restart();
				//	for (int i = 0; i < trials; i++)
				//		_ = detector.DetectPtr(mat, 0.2f);

				//	sw.Stop();
				//	var time = sw.Elapsed;

				//	Console.WriteLine("\nPtr");
				//	Console.WriteLine($"{time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
				//}


				{
					sw = Stopwatch.StartNew();
					for (int i = 0; i < trials; i++)
						_ = detector.DetectRef(mat, 0.2f);

					sw.Stop();
					var time = sw.Elapsed;

					Console.WriteLine("\nRef");
					Console.WriteLine($"{time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
				}
			}
			//}
		}

		//internal void TestDetector(string config, string weights, int gpu, string imageFile)
		//{
		//	IntPtr detector = IntPtr.Zero;
		//	try
		//	{
		//		detector = InteropMethods.CreateDetector(config, weights, gpu);
		//		var mat = Cv2.ImRead(imageFile, ImreadModes.Color);

		//		int count = 0;
		//		while (true)
		//		{

		//			//InteropMethods.BboxContainer container2 = new InteropMethods.BboxContainer();
		//			//int resultsB = InteropMethods.DetectorFromMat(detector, mat.CvPtr, 0.0f, ref container2);

		//			//var resultPtr = InteropMethods.DetectorFromMat(detector, mat.CvPtr, 0.2f);
		//			var resultPtr = InteropMethods.DetectFromFile(detector, imageFile, 0.0f);

		//			var result = Marshal.PtrToStructure<BboxContainer>(resultPtr);
		//			MarshalUnmananagedArrayToStruct<bbox_t>(result.candidatesPtr, result.size, out bbox_t[] candidates);
		//			InteropMethods.DisposeDetections(resultPtr);

		//			if (count++ % 1000 == 0)
		//			{
		//				Console.Write(".");
		//				GC.Collect(2);

		//				//InteropMethods.DisposeDetector(detector);
		//				//detector = InteropMethods.CreateDetector(config, weights, gpu);
		//			}
		//		}
		//	}
		//	catch (Exception ex)
		//	{
		//		InteropMethods.DisposeDetector(detector);
		//	}
		//}

		public static void MarshalUnmananagedArrayToStruct<T>(IntPtr unmanagedArray, long length, out T[] mangagedArray)
		{
			var size = Marshal.SizeOf(typeof(T));
			mangagedArray = new T[length];

			for (int i = 0; i < length; i++)
			{
				IntPtr ins = new IntPtr(unmanagedArray.ToInt64() + i * size);
				if (ins != IntPtr.Zero)
				{
					mangagedArray[i] = Marshal.PtrToStructure<T>(ins);
				}
			}

		}


	}
}
