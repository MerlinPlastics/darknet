#include "interop.hpp"
#include "network.hpp"


extern "C" {
	InteropDetector* CreateInteropDetector(const char* configurationFilename, const char* weightsFilename, int gpu) {
		return new InteropDetector(configurationFilename, weightsFilename, gpu);
	}

	int DisposeInteropDetector(InteropDetector* detector) {
		delete(detector);
		return 1;
	}

	int DetectorDetectMat(InteropDetector* detector, cv::Mat* mat, bbox_t_container& container) {

		std::vector<bbox_t> detection = detector->detect(*mat);
		for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
			container.candidates[i] = detection[i];
		return detection.size();
	}

	int DetectMatInteropDetector(InteropDetector* detector, cv::Mat* mat, float threshold, bbox_t_container& container) {

		std::vector<bbox_t> detections = detector->detect(*mat, threshold);
		container.size = detections.size();

		//container.candidates_ptr = (bbox_t*)xcalloc(detections.size(), sizeof(bbox_t));

		for (size_t i = 0; i < detections.size() && i < C_SHARP_MAX_OBJECTS; ++i){
			container.candidates[i] = detections[i];
			//container.candidates_ptr[i] = detections[i];
		}
		//memcpy(container.candidates_ptr[i], detections.)

		return detections.size();
	}

	int DetectFileInteropDetector(InteropDetector* detector, const char* filename, float threshold, bbox_t_container& container) {
		std::vector<bbox_t> detections = detector->detect(filename, threshold);
		container.size = detections.size();
		for (size_t i = 0; i < detections.size() && i < C_SHARP_MAX_OBJECTS; ++i)
			container.candidates[i] = detections[i];
		return detections.size();
	}
}


InteropDetector::InteropDetector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0)
	: cur_gpu_id(gpu_id)
{

	wait_stream = 0;
#ifdef GPU
	int old_gpu_index;
	check_cuda(cudaGetDevice(&old_gpu_index));
#endif

	detector_gpu_ptr = std::make_shared<detector_gpu_t>();
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());

#ifdef GPU
	cuda_set_device(cur_gpu_id);
	printf(" Used GPU %d \n", cur_gpu_id);
#endif
	network& net = detector_gpu.net;
	net.gpu_index = cur_gpu_id;
	char* cfgfile = const_cast<char*>(cfg_filename.c_str());
	char* weightfile = const_cast<char*>(weight_filename.c_str());

	net = parse_network_cfg_custom(cfgfile, 1, 1);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	net.gpu_index = cur_gpu_id;
	fuse_conv_batchnorm(net);

	layer l = net.layers[net.n - 1];
	int j;

#ifdef GPU
	check_cuda(cudaSetDevice(old_gpu_index));
#endif
}

int InteropDetector::get_net_width() const {
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	return detector_gpu.net.w;
}
int InteropDetector::get_net_height() const {
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	return detector_gpu.net.h;
}
int InteropDetector::get_net_color_depth() const {
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	return detector_gpu.net.c;
}


std::vector<bbox_t> InteropDetector::detect(std::string image_filename, float thresh)
{
	image img = load_image_cv(const_cast<char*>(image_filename.c_str()), 3);
	return detect(img, thresh);
}

//
//image_t InteropDetector::load_image(std::string image_filename)
//{
//	image im = load_image_cv(const_cast<char*>(image_filename.c_str()), 3);
//
//	image_t img;
//	img.c = im.c;
//	img.data = im.data;
//	img.h = im.h;
//	img.w = im.w;
//
//	return img;
//}

//std::vector<bbox_t> InteropDetector::detect(float* data, int size, int width, int height, int channels, float thresh) {
//	image_t image;
//
//	image.w = width;
//	image.h = height;
//	image.c = channels;
//	image.data = (float*)calloc(;
//
//	detect(image); 
//
//}


std::vector<bbox_t> InteropDetector::detect(cv::Mat mat, float thresh)
{
	if (mat.data == NULL)
		throw std::runtime_error("Image is empty");

	cv::Mat img;
	if (mat.channels() == 4) cv::cvtColor(mat, img, cv::COLOR_RGBA2BGR);
	else if (mat.channels() == 3) cv::cvtColor(mat, img, cv::COLOR_RGB2BGR);
	else if (mat.channels() == 1) cv::cvtColor(mat, img, cv::COLOR_GRAY2BGR);

	auto image_ptr = mat_to_image_resize(img);
	// Mat has its original size
	return detect_resized(image_ptr, mat.cols, mat.rows, thresh);
}
//
//std::vector<bbox_t> InteropDetector::detect(image_t img, float thresh)
//{
//	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
//	network& net = detector_gpu.net;
//#ifdef GPU
//	int old_gpu_index;
//	cudaGetDevice(&old_gpu_index);
//	if (cur_gpu_id != old_gpu_index)
//		cudaSetDevice(net.gpu_index);
//
//	net.wait_stream = wait_stream;    // 1 - wait CUDA-stream, 0 - not to wait
//#endif
//
//	image im;
//	im.c = img.c;
//	im.data = img.data;
//	im.h = img.h;
//	im.w = img.w;
//
//	image sized;
//
//	if (net.w == im.w && net.h == im.h) {
//		sized = make_image(im.w, im.h, im.c);
//		memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
//	}
//	else
//		sized = resize_image(im, net.w, net.h);
//
//	layer outputLayer = net.layers[net.n - 1];
//
//	float* X = sized.data;
//
//	float* prediction = network_predict(net, X);
//
//	int nboxes = 0;
//	int letterbox = 0;
//	float hier_thresh = 0.5;
//	detection* dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
//	if (nms) do_nms_sort(dets, nboxes, outputLayer.classes, nms);
//
//	std::vector<bbox_t> bbox_vec;
//
//	for (int i = 0; i < nboxes; ++i) {
//		box b = dets[i].bbox;
//		int const obj_id = max_index(dets[i].prob, outputLayer.classes);
//		float const prob = dets[i].prob[obj_id];
//
//		if (prob > thresh)
//		{
//			bbox_t bbox;
//			bbox.x = std::max((double)0, (b.x - b.w / 2.) * im.w);
//			bbox.y = std::max((double)0, (b.y - b.h / 2.) * im.h);
//			bbox.w = b.w * im.w;
//			bbox.h = b.h * im.h;
//			bbox.class_id = obj_id;
//			bbox.confidence = prob;
//
//			bbox_vec.push_back(bbox);
//		}
//	}
//
//	free_detections(dets, nboxes);
//	if (sized.data)
//		free(sized.data);
//
//#ifdef GPU
//	if (cur_gpu_id != old_gpu_index)
//		cudaSetDevice(old_gpu_index);
//#endif
//
//	return bbox_vec;
//}

std::vector<bbox_t> InteropDetector::detect(image img, float thresh)
{
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	network& net = detector_gpu.net;
#ifdef GPU
	int old_gpu_index;
	cudaGetDevice(&old_gpu_index);
	if (cur_gpu_id != old_gpu_index)
		cudaSetDevice(net.gpu_index);

	net.wait_stream = wait_stream;    // 1 - wait CUDA-stream, 0 - not to wait
#endif

	image sized;

	if (net.w == img.w && net.h == img.h) {
		sized = make_image(img.w, img.h, img.c);
		memcpy(sized.data, img.data, img.w * img.h * img.c * sizeof(float));
	}
	else
		sized = resize_image(img, net.w, net.h);

	layer outputLayer = net.layers[net.n - 1];

	float* X = sized.data;

	float* prediction = network_predict(net, X);

	int nboxes = 0;
	int letterbox = 0;
	float hier_thresh = 0.5;
	detection* dets = get_network_boxes(&net, img.w, img.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
	if (nms) do_nms_sort(dets, nboxes, outputLayer.classes, nms);

	std::vector<bbox_t> bbox_vec;

	for (int i = 0; i < nboxes; ++i) {
		box b = dets[i].bbox;
		int const obj_id = max_index(dets[i].prob, outputLayer.classes);
		float const prob = dets[i].prob[obj_id];

		if (prob > thresh)
		{
			bbox_t bbox;
			bbox.x = std::max((double)0, (b.x - b.w / 2.) * img.w);
			bbox.y = std::max((double)0, (b.y - b.h / 2.) * img.h);
			bbox.w = b.w * img.w;
			bbox.h = b.h * img.h;
			bbox.class_id = obj_id;
			bbox.confidence = prob;

			bbox_vec.push_back(bbox);
		}
	}

	free_detections(dets, nboxes);
	if (sized.data)
		free(sized.data);

#ifdef GPU
	if (cur_gpu_id != old_gpu_index)
		cudaSetDevice(old_gpu_index);
#endif

	return bbox_vec;
}

InteropDetector::~InteropDetector()
{
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());

#ifdef GPU
	int old_gpu_index;
	cudaGetDevice(&old_gpu_index);
	cuda_set_device(detector_gpu.net.gpu_index);
#endif

	free_network(detector_gpu.net);

#ifdef GPU
	cudaSetDevice(old_gpu_index);
#endif
}



