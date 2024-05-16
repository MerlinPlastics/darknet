using System.Runtime.InteropServices;

namespace DarknetDotnet
{
	internal class InteropMethods
	{
		internal struct bbox_t
		{
			public uint x;

			public uint y;

			public uint w;

			public uint h;

			public float confidence;

			public uint obj_id;

		}

		[StructLayout(LayoutKind.Sequential)]
		internal struct BboxContainer
		{
			public long size;

			public IntPtr candidatesPtr;
		}


		private const string DarknetLibraryName = @"x64\darknet.dll";

		
		[DllImport(DarknetLibraryName, EntryPoint = "CreateInteropDetector")]
		public static extern IntPtr CreateDetector(string configurationFilename, string weightsFilename, int gpu);


		[DllImport(DarknetLibraryName, EntryPoint = "SpeedInteropDetector")]
		public static extern double SpeedTest(IntPtr detector, int trials);


		[DllImport(DarknetLibraryName, EntryPoint = "DetectMatInteropDetector")]
		public static extern IntPtr DetectFromMat(IntPtr detector, IntPtr mat, float threshold);


		[DllImport(DarknetLibraryName, EntryPoint = "DetectFileInteropDetector")]
		public static extern IntPtr DetectFromFile(IntPtr detector, string filename, float threshold);


		[DllImport(DarknetLibraryName, EntryPoint = "GetDimensionsInteropDetector")]
		public static extern int DetectorSizes(IntPtr detector, ref int width, ref int height, ref int channels);


		[DllImport(DarknetLibraryName, EntryPoint = "DisposeInteropDetector")]
		public static extern int DisposeDetector(IntPtr detector);


		[DllImport(DarknetLibraryName, EntryPoint = "DisposeContainerInteropDetector")]
		public static extern int DisposeDetections(IntPtr detections);

	}
}
