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

		[StructLayout(LayoutKind.Sequential)]
		internal struct BboxContainerRef
		{
			public long size;

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 1000)]
			public bbox_t[] candidates;
		}


		private const string DarknetLibraryName = @"x64\darknet.dll";

		
		[DllImport(DarknetLibraryName, EntryPoint = "CreateInteropDetector")]
		public static extern IntPtr CreateDetector(string configurationFilename, string weightsFilename, int gpu);


		[DllImport(DarknetLibraryName, EntryPoint = "SpeedInteropDetector")]
		public static extern double SpeedTest(IntPtr detector, int trials);

		// =============================================================================
		// Detection with pointer-based output
		[DllImport(DarknetLibraryName, EntryPoint = "DetectFileInteropDetectorPtr")]
		public static extern IntPtr DetectFromFilePtr(IntPtr detector, string filename, float threshold);

		[DllImport(DarknetLibraryName, EntryPoint = "DetectMatInteropDetectorPtr")]
		public static extern IntPtr DetectFromMatPtr(IntPtr detector, IntPtr mat, float threshold);


		// Detection with array reference based output
		[DllImport(DarknetLibraryName, EntryPoint = "DetectFileInteropDetectorRef")]
		public static extern int DetectFromFileRef(IntPtr detector, string filename, float threshold, ref BboxContainerRef container);

		[DllImport(DarknetLibraryName, EntryPoint = "DetectMatInteropDetectorRef")]
		public static extern int DetectFromMatRef(IntPtr detector, IntPtr mat, float threshold, ref BboxContainerRef container);


		// =============================================================================
		[DllImport(DarknetLibraryName, EntryPoint = "GetDimensionsInteropDetector")]
		public static extern int DetectorSizes(IntPtr detector, ref int width, ref int height, ref int channels);


		[DllImport(DarknetLibraryName, EntryPoint = "DisposeInteropDetector")]
		public static extern int DisposeDetector(IntPtr detector);


		[DllImport(DarknetLibraryName, EntryPoint = "DisposeContainerInteropDetector")]
		public static extern int DisposeDetections(IntPtr detections);

	}
}
