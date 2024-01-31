#include "image.hpp"
#include "image_opencv.hpp"
#include "darknetdotnet.hpp"

using namespace cv;
using namespace System;

#define CV_8UC5 CV_MAKETYPE(CV_8U,5)
#define CV_8UC6 CV_MAKETYPE(CV_8U,6)
#define CV_8UC7 CV_MAKETYPE(CV_8U,7)
#define CV_8UC8 CV_MAKETYPE(CV_8U,8)

darknetdotnet::Detector::Detector(std::string configFile, std::string weightFile)
{
	/*
	detector_gpu_ptr = std::make_shared<detector_gpu_t>();
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());

	_configFile = configFile;
	_weightFile = weightFile;
	*/
}

bool darknetdotnet::Detector::Test()
{
	try {
		//auto img = load_image("C:\\Temp\\barcode.png", 0, 0, 0);
		Mat mat(240, 240, CV_8UC5);
		std::vector<Mat> imgs;
		imgs.push_back(mat);
		imgs.push_back(~mat);
		//imgs.push_back(mat(Rect(0, 0, mat.cols / 2, mat.rows / 2)));
		imwrite("test.tiff", imgs);

		return true;
	}
	catch (System::Exception^ ex) {
		throw ex;
	}

	return false;
}
