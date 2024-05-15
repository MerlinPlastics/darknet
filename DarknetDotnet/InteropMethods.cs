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

			[MarshalAs(UnmanagedType.ByValArray, SizeConst = 1000)]
			public bbox_t[] candidates;

			public IntPtr candidatesPtr;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct DarknetImage
		{
			public int w;
			public int h;
			public int c;
			public IntPtr data;
		}

		private const string DarknetLibraryName = @"..\..\..\..\build\src-lib\Debug\darknet.dll";

		private const int MaxObjects = 1000;

		[DllImport(DarknetLibraryName, EntryPoint = "CreateInteropDetector")]
		public static extern IntPtr CreateDetector(string configurationFilename, string weightsFilename, int gpu);


		//[DllImport(DarknetLibraryName, EntryPoint = "DetectorDetectBuffer")]
		//private static extern int DetectFromBuffer(IntPtr detector, IntPtr pArray, int nSize, ref BboxContainer container);

		[DllImport(DarknetLibraryName, EntryPoint = "DetectMatInteropDetector", CallingConvention = CallingConvention.StdCall)]
		public static extern int DetectorFromMat(IntPtr detector, IntPtr matPtr, float threshold, ref BboxContainer container);

		//[DllImport(DarknetLibraryName, EntryPoint = "DetectMatInteropDetectorPtr")]
		//public static extern IntPtr DetectMatInteropDetectorPtr(IntPtr detector, IntPtr mat, float threshold, out int size);


		[DllImport(DarknetLibraryName, EntryPoint = "DetectFileInteropDetector")]
		public static extern int DetectorFromFile(IntPtr detector, string filename, float threshold, ref BboxContainer container);

		[DllImport(DarknetLibraryName, EntryPoint = "GetDimensionsInteropDetector")]
		public static extern int DetectorSizes(IntPtr detector, ref int width, ref int height, ref int channels);


		[DllImport(DarknetLibraryName, EntryPoint = "DisposeInteropDetector")]
		public static extern int DisposeDetector(IntPtr detector);

		[DllImport(DarknetLibraryName, EntryPoint = "DisposeDetectionsInteropDetector")]
		public static extern int DisposeDetections(IntPtr detections, int size);


		//[DllImport(DarknetLibraryName, EntryPoint = "load_network_custom")]
		//public static extern IntPtr LoadNetwork(string configurationFilename, string weightsFilename, int gpu,
		//	int batch);

		//[DllImport(DarknetLibraryName, EntryPoint = "free_network_ptr")]
		//public static extern void FreeNetwork(IntPtr networkPtr);

		//[DllImport(DarknetLibraryName, EntryPoint = "network_predict_image_ptr")]
		//public static extern IntPtr PredictImage(IntPtr network, IntPtr image, out int outputLength);

		//[DllImport(DarknetLibraryName, EntryPoint = "load_image_ptr")]
		//public static extern IntPtr LoadImagePtr(int desired_width, int desired_height, int channels, string filename);


		//[DllImport(DarknetLibraryName, EntryPoint = "load_image")]
		//public static extern IntPtr LoadImage(string filename, int desired_width, int desired_height, int channels);

		//[DllImport(DarknetLibraryName, EntryPoint = "free_image")]
		//public static extern void FreeImage(IntPtr image);

		//[DllImport(DarknetLibraryName, EntryPoint = "network_width")]
		//public static extern int NetworkWidth(IntPtr network);

		//[DllImport(DarknetLibraryName, EntryPoint = "network_height")]
		//public static extern int NetworkHeight(IntPtr network);
	}
}
