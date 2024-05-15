#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "darknet.h"

//extern "C" {
#include "parser.hpp"
#include "image_opencv.hpp"
//}
#include "Timing.hpp"


#define C_SHARP_MAX_OBJECTS 1000

struct bbox_t {
	unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box as % of input image
	float confidence;              // confidence - probability that the object was found correctly
	unsigned int class_id;         // class of object - from range [0, classes-1]
};

struct bbox_t_container {
	size_t size;
	bbox_t* candidates_ptr;
	//bbox_t candidates[C_SHARP_MAX_OBJECTS];
};


struct detector_gpu_t {
	network net;
	float* predictions;
};



class InteropDetector
{
	std::shared_ptr<void> detector_gpu_ptr;
	const int cur_gpu_id;

public:
	InteropDetector(std::string cfg_filename, std::string weight_filename, int gpu_id);
	~InteropDetector();

	std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2);
	std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2);
	std::vector<bbox_t> detect(image image, float thresh = 0.2);

	int get_net_width() const;
	int get_net_height() const;
	int get_net_channels() const;

private:

	float nms = .4;
	bool wait_stream;

	std::vector<bbox_t> detect_resized(image img, int init_w, int init_h, float thresh = 0.2)
	{
		if (img.data == NULL)
			throw std::runtime_error("Image is empty");
		auto detection_boxes = detect(img, thresh);

		// Rescale the output values to the original input image
		float wk = (float)init_w / img.w, hk = (float)init_h / img.h;
		for (auto& i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;

		return detection_boxes;
	}

private:
	image mat_to_image_resize(cv::Mat mat) const
	{
		image image;
		if (mat.data == NULL) return image;

		cv::Size network_size = cv::Size(get_net_width(), get_net_height());
		cv::Mat det_mat;

		cv::resize(mat, det_mat, network_size);

		if (det_mat.channels() == 4) cv::cvtColor(det_mat, det_mat, cv::COLOR_RGBA2BGR);
		else if (det_mat.channels() == 3) cv::cvtColor(det_mat, det_mat, cv::COLOR_RGB2BGR);
		else if (det_mat.channels() == 1) cv::cvtColor(det_mat, det_mat, cv::COLOR_GRAY2BGR);
		else std::cerr << " Warning: img_src.channels() is not 1, 3 or 4. It is = " << det_mat.channels() << std::endl;

		image = mat_to_image(det_mat);

		return image;
	}

private:

	void check_cuda(cudaError_t status) {
		if (status != cudaSuccess) {
			const char* s = cudaGetErrorString(status);
			printf("CUDA Error Prev: %s\n", s);
		}
	}

	void* get_cuda_context()
	{
#ifdef GPU
		int old_gpu_index;
		cudaGetDevice(&old_gpu_index);
		if (cur_gpu_id != old_gpu_index)
			cudaSetDevice(cur_gpu_id);

		void* cuda_context = cuda_get_context();

		if (cur_gpu_id != old_gpu_index)
			cudaSetDevice(old_gpu_index);

		return cuda_context;
#else   // GPU
		return NULL;
#endif  // GPU
	}

};


