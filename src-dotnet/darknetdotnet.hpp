#pragma once

//#include "image.hpp"
//#include "image_opencv.hpp"

using namespace System;

namespace darknetdotnet {
	std::shared_ptr<void> detector_gpu_ptr;

	public ref class Detector {
	public:
		// constructor
		Detector(std::string configFile, std::string weightFile);
	
		bool Test();
	};

}

