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
			string config = "cfg-y4-smaller.cfg";
			string weights = "cfg-y4-smaller_final.weights";
			string names = "cfg-y4-smaller.names";

			//string config = "calset.cfg";
			//string weights = "calset_final.weights";
			//string names = "calset.names";

			int gpu = 0;

			//string imageFile = @"image.jpg";

			Program p = new Program();

			//int trials = Int32.Parse(args[0]);
			//int concurrent = Int32.Parse(args[1]);
			//p.TestDetector(config, weights, names, gpu, imageFile, trials,concurrent);

			//p.TestNetworkBoxes(config, weights, names, gpu, imageFile);
			p.TestValidator(config, weights, names, gpu);
		}

		internal void TestDetector(string config, string weights, string names, int gpu, string imageFile, int trials, int concurrent)
		{
			var mat = Cv2.ImRead(imageFile, ImreadModes.Color);

			var sem = new Semaphore(0, concurrent);
			List<Task> tasks = new List<Task>();
			Parallel.For(0, concurrent, i =>
			{
				var t = Task.Run(() =>
				{
					using (var detector = YoloDetector.CreateDetector(config, weights, names, 0))
					{
						int width, height, colors;
						(width, height, colors) = detector.GetNetworkDimensions();

						mat = new Mat(height, width, MatType.CV_8UC3);

						Console.WriteLine($"{i}: waiting");
						sem.WaitOne(0);

						//int trials = 100;
						Stopwatch sw = Stopwatch.StartNew();

						{
							sw.Restart();
							_ = detector.SpeedTest(trials);
							sw.Stop();

							var time = sw.Elapsed;

							Console.WriteLine("SpeedTest");
							Console.WriteLine(
								$"{i}: {time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
							Console.WriteLine();
						}

						{
							sw.Restart();
							for (int i = 0; i < trials; i++)
								_ = detector.Detect(mat, 0.2f);

							sw.Stop();
							var time = sw.Elapsed;

							Console.WriteLine("YoloDetector with pointer");
							Console.WriteLine(
								$"{i}: {time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
							Console.WriteLine();
						}

						{
							sw.Restart();
							for (int i = 0; i < trials; i++)
								_ = detector.Detect(mat, 0.2f);

							sw.Stop();
							var time = sw.Elapsed;

							Console.WriteLine("YoloDetector with reference array");
							Console.WriteLine(
								$"{i}: {time.TotalMilliseconds:0.00} ms for {trials} trials, or {trials * 1000.0 / time.TotalMilliseconds:0.0} Hz");
							Console.WriteLine();
						}

						Console.WriteLine($"{i}: done");
					}
				});
				tasks.Add(t);
			});

			sem.Release(6);
			Task.WaitAll(tasks.ToArray());
		}

		internal void TestNetworkBoxes(string config, string weights, string names, int gpu, string imageFile)
		{
			var mat = Cv2.ImRead(imageFile, ImreadModes.Color);
			using (var detector = YoloDetector.CreateDetector(config, weights, names, 0))
			{
				int width, height, colors;
				(width, height, colors) = detector.GetNetworkDimensions();

				//mat = new Mat(height, width, MatType.CV_8UC3);
				var results = detector.Detect(mat, 0.2f);
				var detections = detector.GetNetworkBoxes(mat, 0.2f);
			}
		}

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

		public void TestValidator(string config, string weights, string names, int gpu)
		{
			var validator = new Validator(config, weights, names, gpu);
			var imageFile = "mixed.png";
			var truthsFile = "mixed.txt";

			var stats = validator.ValidateImage(".", imageFile, truthsFile, 0.5f, 0.25f);

			validator.PrintStats(stats);
		}
	}
}
