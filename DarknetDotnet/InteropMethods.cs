using System.Runtime.InteropServices;

namespace DarknetDotnet
{
	internal class InteropMethods
	{
		public struct bbox_t
		{
			public uint x;

			public uint y;

			public uint w;

			public uint h;

			public float confidence;

			public uint obj_id;

		}

		// Interop structure.  Assume we won't have more than 1000 detections on a single frame
		[StructLayout(LayoutKind.Sequential)]
		public struct BboxContainer
		{
			public long size;

			public IntPtr candidatesPtr;

			//[MarshalAs(UnmanagedType.ByValArray, SizeConst = 1000)]
			//public bbox_t[] candidates;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct DarknetImage
		{
			public int w;
			public int h;
			public int c;
			public IntPtr data;
		}

		private const string DarknetLibraryName = @"x64\darknet.dll";

		private const int MaxObjects = 1000;

		[DllImport(DarknetLibraryName, EntryPoint = "CreateInteropDetector")]
		public static extern IntPtr CreateDetector(string configurationFilename, string weightsFilename, int gpu);

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
