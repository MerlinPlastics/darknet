using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static DarknetDotnet.InteropMethods;

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

			var resultPtr = InteropMethods.DetectFromMat(detector, mat.CvPtr, threshold);
			return ConvertUnmanagedBBoxContainer(resultPtr);
		}

		public IEnumerable<YoloItem> Detect(string fileName, float threshold = 0.2f)
		{
			if (!File.Exists(fileName))
				return Enumerable.Empty<YoloItem>();

			var resultPtr = InteropMethods.DetectFromFile(detector, fileName, threshold);
			return ConvertUnmanagedBBoxContainer(resultPtr);
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

	}
}
