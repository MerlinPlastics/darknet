#include "interop.hpp"
#include "network.hpp"

/// <summary>
/// High level InteropDetector object for quick and easy inference
/// </summary>
extern "C" {
	InteropDetector* CreateInteropDetector(const char* configurationFilename, const char* weightsFilename, int gpu) {
		return new InteropDetector(configurationFilename, weightsFilename, gpu);
	}

	int GetDimensionsInteropDetector(InteropDetector* detector, int& width, int& height, int& channels) {
		if (detector == NULL)
			return 0;

		width = detector->get_net_width();
		height = detector->get_net_height();
		channels = detector->get_net_channels();

		return 1;
	}

	int DisposeInteropDetector(InteropDetector* detector) {
		if (detector)
			delete(detector);

		return 1;
	}


	bbox_t_container* DetectFileInteropDetector(InteropDetector* detector, const char* filename, float threshold) {
		bbox_t_container* container = (bbox_t_container*)xmalloc(sizeof(bbox_t_container));

		std::vector<bbox_t> detections = detector->detect(filename, threshold);

		container->size = detections.size();
		container->candidates_ptr = (bbox_t*)xcalloc(container->size, sizeof(bbox_t));

		for (size_t i = 0; i < detections.size(); ++i) {
			//container->candidates[i] = detections[i];
			container->candidates_ptr[i] = detections[i];
		}

		return container;
	}

	bbox_t_container* DetectMatInteropDetector(InteropDetector* detector, cv::Mat* mat, float threshold) {

		bbox_t_container* container = (bbox_t_container*)xmalloc(sizeof(bbox_t_container));

		std::vector<bbox_t> detections = detector->detect(*mat, threshold);
		container->size = detections.size();
		container->candidates_ptr = (bbox_t*)xcalloc(container->size, sizeof(bbox_t));

		for (size_t i = 0; i < detections.size(); ++i) {
			//container->candidates[i] = detections[i];
			container->candidates_ptr[i] = detections[i];
		}

		return container;
	}

	int DisposeContainerInteropDetector(bbox_t_container* container) {

		if (container->candidates_ptr) {
			free(container->candidates_ptr);
		}

		free(container);

		return 1;
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
int InteropDetector::get_net_channels() const {
	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	return detector_gpu.net.c;
}


std::vector<bbox_t> InteropDetector::detect(std::string image_filename, float thresh)
{
	image img = load_image_cv(const_cast<char*>(image_filename.c_str()), 3);
	auto results = detect(img, thresh);

	free_image(img);

	return results;
}


std::vector<bbox_t> InteropDetector::detect(cv::Mat mat, float thresh)
{
	if (mat.data == NULL)
		throw std::runtime_error("Image is empty");

	auto image_ptr = mat_to_image_resize(mat);

	// Mat has its original size
	auto results = detect_resized(image_ptr, mat.cols, mat.rows, thresh);
	//std::vector<bbox_t> results;

	free_image(image_ptr);

	return results;
}



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



