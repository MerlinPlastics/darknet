using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static DarknetDotnet.InteropMethods;
using Size = System.Drawing.Size;

namespace DarknetDotnet
{
	public class YoloDetector : IDisposable
	{

		private IntPtr detector;
		private bool disposedValue = false;
		private Dictionary<int, string> classNames;

		public YoloDetector(string configFile, string weightsFile, int gpu, Dictionary<int, string> classNames)
		{
			this.detector = InteropMethods.CreateDetector(configFile, weightsFile, gpu);
			this.classNames = classNames;
		}

		public IEnumerable<YoloItem> Detect(Mat mat, float threshold = 0.2f)
		{
			if (mat == null || mat.CvPtr == IntPtr.Zero)
				return Enumerable.Empty<YoloItem>();

			var resultPtr = InteropMethods.DetectFromMatPtr(detector, mat.CvPtr, threshold);
			return ConvertUnmanagedBBoxContainer(resultPtr);
		}

		public IEnumerable<YoloItem> Detect(string fileName, float threshold = 0.2f)
		{
			if (!File.Exists(fileName))
				return Enumerable.Empty<YoloItem>();

			var resultPtr = InteropMethods.DetectFromFilePtr(detector, fileName, threshold);
			return ConvertUnmanagedBBoxContainer(resultPtr);
		}

		public TimeSpan SpeedTest(int trials = 1000)
		{
			var ms = InteropMethods.SpeedTest(detector, trials);
			return TimeSpan.FromMilliseconds(ms);
		}

		public (int, int, int) GetNetworkDimensions()
		{
			int width = 0;
			int height = 0;
			int channels = 0;
			_ = InteropMethods.DetectorSizes(detector, ref width, ref height, ref channels);

			return (width, height, channels);
		}

		private IEnumerable<YoloItem> ConvertUnmanagedBBoxContainer(nint resultPtr)
		{

			// Marshal the BBoxContainer, then convert the unmanaged candidates array pointer
			// to an array of managed structs
			var container = Marshal.PtrToStructure<BboxContainer>(resultPtr);
			MarshalUnmananagedArrayToStruct<bbox_t>(container.candidatesPtr, container.size, out bbox_t[] candidates);
			InteropMethods.DisposeDetections(resultPtr);

			// Convert the array to our objects
			List<YoloItem> items = new List<YoloItem>();
			foreach (var box in candidates.Where(b => b.h > 0 && b.w > 0))
			{
				var yoloItem = new YoloItem()
				{
					X = (int)box.x,
					Y = (int)box.y,
					Width = (int)box.w,
					Height = (int)box.h,
					Confidence = box.confidence,
					ObjectTypeId = (int)box.obj_id
				};

				if (this.classNames.TryGetValue((int)box.obj_id, out string? typeName))
					yoloItem.Type = typeName;

				items.Add(yoloItem);
			}

			return items;
		}

		protected virtual void Dispose(bool disposing)
		{
			if (!disposedValue)
			{
				if (disposing)
				{
					// TODO: dispose managed state (managed objects)
				}

				// TODO: free unmanaged resources (unmanaged objects) and override finalizer
				if (detector != IntPtr.Zero)
				{
					InteropMethods.DisposeDetector(detector);
				}
				// TODO: set large fields to null
				disposedValue = true;
			}
		}

		// TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
		~YoloDetector()
		{
			// Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
			Dispose(disposing: false);
		}

		public void Dispose()
		{
			// Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
			Dispose(disposing: true);
			GC.SuppressFinalize(this);
		}

		static void MarshalUnmananagedArrayToStruct<T>(IntPtr unmanagedArray, long length, out T[] mangagedArray)
		{
			var size = Marshal.SizeOf(typeof(T));
			mangagedArray = new T[length];

			for (int i = 0; i < length; i++)
			{
				IntPtr ins = new IntPtr(unmanagedArray.ToInt64() + i * size);
				if (ins != IntPtr.Zero)
				{
#pragma warning disable CS8601 // Possible null reference assignment.
					mangagedArray[i] = Marshal.PtrToStructure<T>(ins);
#pragma warning restore CS8601 // Possible null reference assignment.
				}
			}

		}

		public static YoloDetector CreateDetector(string configurationFile, string weightsFile, string namesFile,
			int gpu)
		{
			if (configurationFile == null || !Path.Exists(configurationFile))
			{
				throw new ArgumentNullException($"No .cfg file found at {configurationFile}");
			}

			if (weightsFile == null || !Path.Exists(weightsFile))
			{
				throw new ArgumentNullException($"No .weights file found at {weightsFile}");
			}

			if (namesFile == null || !Path.Exists(namesFile))
			{
				throw new ArgumentNullException($"No .names file found at {namesFile}");
			}

			try
			{
				int i = 0;
				var classes = File.ReadAllLines(namesFile)
					.Select(q => q.ToLower().Trim())
					.Where(q => !string
						.IsNullOrEmpty(q))
					.ToDictionary(value => i++, value => value);


				return new YoloDetector(configurationFile, weightsFile, gpu, classes);
			}
			catch (Exception)
			{
				throw;
			}
		}

		public static YoloDetector CreateDetector(string? neuralNetworkPath, int gpu)
		{
			if (!Path.IsPathRooted(neuralNetworkPath))
			{
				string currentPath = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)!;

				if (!string.IsNullOrEmpty(neuralNetworkPath))
					neuralNetworkPath = Path.GetFullPath(Path.Combine(currentPath, neuralNetworkPath));
				else
					neuralNetworkPath = currentPath;
			}

			string yoloPath = $"{neuralNetworkPath}";

			if (!Directory.Exists(yoloPath))
			{
				throw new InvalidOperationException($"Cannot find neural network path at {yoloPath} ");
			}

			// Find all files in target directory
			var files = Directory.GetFiles(yoloPath, "*.*", SearchOption.TopDirectoryOnly);

			var configurationFile = files.Where(o => o.EndsWith(".cfg")).FirstOrDefault();
			var weightsFile = files.Where(o => o.EndsWith(".weights")).FirstOrDefault();
			var namesFile = files.Where(o => o.EndsWith(".names")).FirstOrDefault();

			if (configurationFile == null)
			{
				throw new ArgumentNullException($"No .cfg file found at {yoloPath}");
			}

			if (weightsFile == null)
			{
				throw new ArgumentNullException($"No .weights file found at {yoloPath}");
			}

			if (namesFile == null)
			{
				throw new ArgumentNullException($"No .names file found at {yoloPath}");
			}

			try
			{
				int i = 0;
				var classes = File.ReadAllLines(namesFile)
					.Select(q => q.ToLower().Trim())
					.Where(q => !string
						.IsNullOrEmpty(q))
					.ToDictionary(value => i++, value => value);


				return new YoloDetector(configurationFile, weightsFile, gpu, classes);
			}
			catch (Exception)
			{
				throw;
			}
		}



		public IEnumerable<YoloItem> DetectRef(Mat mat, float threshold = 0.2f)
		{
			if (mat == null || mat.CvPtr == IntPtr.Zero)
				return Enumerable.Empty<YoloItem>();


			BboxContainerRef container = default(BboxContainerRef);
			var count = InteropMethods.DetectFromMatRef(detector, mat.CvPtr, threshold, ref container);

			return Convert(container);

		}

		public IEnumerable<YoloItem> DetectRef(string fileName, float threshold = 0.2f)
		{
			if (!File.Exists(fileName))
				return Enumerable.Empty<YoloItem>();

			BboxContainerRef container = default(BboxContainerRef);
			var resultPtr = InteropMethods.DetectFromFileRef(detector, fileName, threshold, ref container);

			return Convert(container);
		}


		private List<YoloItem> Convert(BboxContainerRef container, float? confidence = null)
		{
			var items = new List<YoloItem>();
			foreach (var box in container.candidates.Where(o => o.h > 0 && o.w > 0))
			{
				var yoloItem = new YoloItem
				{
					X = (int)box.x,
					Y = (int)box.y,
					Height = (int)box.h,
					Width = (int)box.w,
					Confidence = box.confidence,
					ObjectTypeId = (int)box.obj_id,
				};

				if (this.classNames.TryGetValue((int)box.obj_id, out string? typeName))
					yoloItem.Type = typeName;

				items.Add(yoloItem);
			}

			return items;
		}
	}
}
