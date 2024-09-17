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

	double SpeedInteropDetector(InteropDetector* detector, int trials) {
		double time = detector->speed(trials);

		return time;
	}

	int DisposeInteropDetector(InteropDetector* detector) {
		if (detector)
			delete(detector);

		return 1;
	}


	bbox_t_container_ptr* DetectFileInteropDetectorPtr(InteropDetector* detector, const char* filename, float threshold) {
		bbox_t_container_ptr* container = (bbox_t_container_ptr*)xmalloc(sizeof(bbox_t_container_ptr));

		std::vector<bbox_t> detections = detector->detect(filename, threshold);

		container->size = detections.size();
		container->candidates_ptr = (bbox_t*)xcalloc(container->size, sizeof(bbox_t));

		for (size_t i = 0; i < detections.size(); ++i) {
			container->candidates_ptr[i] = detections[i];
		}

		return container;
	}

	bbox_t_container_ptr* DetectMatInteropDetectorPtr(InteropDetector* detector, cv::Mat* mat, float threshold) {

		bbox_t_container_ptr* container = (bbox_t_container_ptr*)xmalloc(sizeof(bbox_t_container_ptr));

		std::vector<bbox_t> detections = detector->detect(*mat, threshold);
		container->size = detections.size();
		container->candidates_ptr = (bbox_t*)xcalloc(container->size, sizeof(bbox_t));

		for (size_t i = 0; i < detections.size(); ++i) {
			container->candidates_ptr[i] = detections[i];
		}

		return container;
	}

	detection_t_container_ptr* DetectNetworkBoxesInteropDetectorPtr(InteropDetector* detector, cv::Mat* mat, float threshold) {

		//printf("DetectNetworkBoxesInteropDetectorPtr\n");
		detection_t_container_ptr* container = (detection_t_container_ptr*)xmalloc(sizeof(detection_t_container_ptr));
		container->size = 0;

		//printf("getting boxes\n");
		std::vector<mydetection_t> detections = detector->getnetworkboxes(*mat, threshold);

		//printf("Got boxes: %l\n", detections.size());
		container->size = detections.size();

		container->detections_ptr = (mydetection_t*)xcalloc(container->size, sizeof(mydetection_t));

		// Copy the detections to the container
		for (size_t i = 0; i < container->size; i++) {
			//printf("** Copying over %i with classes %i\n", i, detections[i].classes);
			container->detections_ptr[i] = detections[i];
		}

		return container;
	}



	int DetectFileInteropDetectorRef(InteropDetector* detector, const char* filename, float threshold, bbox_t_container& container) {

		std::vector<bbox_t> detections = detector->detect(filename, threshold);

		for (size_t i = 0; i < detections.size(); ++i) {
			container.candidates[i] = detections[i];
		}
		container.size = detections.size();

		return detections.size();
	}



	int DetectMatInteropDetectorRef(InteropDetector* detector, cv::Mat* mat, float threshold, bbox_t_container& container) {

		std::vector<bbox_t> detections = detector->detect(*mat, threshold);

		for (size_t i = 0; i < detections.size(); ++i) {
			container.candidates[i] = detections[i];
		}

		return detections.size();
	}


	int DisposeBBoxContainerInteropDetector(bbox_t_container_ptr* container) {

		if (container->candidates_ptr) {
			free(container->candidates_ptr);
		}

		free(container);

		return 1;
	}

	int DisposeDetectionsContainerInteropDetector(detection_t_container_ptr* container) {

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


std::vector<mydetection_t> InteropDetector::getnetworkboxes(cv::Mat mat, float thresh) {
	if (mat.data == NULL)
		throw std::runtime_error("Image is empty");

	auto image_ptr = mat_to_image_resize(mat);

	// Mat has its original size
	auto results = getnetworkboxes(image_ptr, thresh);

	free_image(image_ptr);

	return results;
}


std::vector<mydetection_t> InteropDetector::getnetworkboxes(image img, float thresh)
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
	//if (nms) do_nms_sort(dets, nboxes, outputLayer.classes, nms);

	std::vector<mydetection_t> detection_vec;
	detection_vec.reserve(nboxes);

	for (int i = 0; i < nboxes; ++i) {
		printf("Doing %i\n", i);

		if (dets[i].objectness == 0)
			continue;

		mydetection_t det;
		
		box b = dets[i].bbox;
		det.x = b.x;
		det.y = b.y;
		det.w = b.w;
		det.h = b.h;

		det.classes = dets[i].classes;
		det.objectness = dets[i].objectness;

		//printf("Doing %i probs with %i classes\n", i, dets[i].classes);
		//det.probs = (float *)xcalloc(C_SHARP_MAX_OBJECTS, sizeof(float));

		// Copy over the probabilities
		memcpy(det.probs, dets[i].prob, dets[i].classes * sizeof(float));


		// Resize the vector to hold 'classes' number of probabilities
		//det.probs.resize(dets[i].classes);

		//// Correctly copy the values from dets[i].prob to det.probs
		//std::copy(dets[i].prob, dets[i].prob + dets[i].classes, det.probs.begin());

		printf("Doing %i adding to main\n", i);
		detection_vec.push_back(det);
	}

	free_detections(dets, nboxes);
	if (sized.data)
		free(sized.data);

#ifdef GPU
	if (cur_gpu_id != old_gpu_index)
		cudaSetDevice(old_gpu_index);
#endif


	return detection_vec;
}


double InteropDetector::speed(int trials)
{

	if (trials <= 0) trials = 1000;

	detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
	network& net = detector_gpu.net;

	set_batch_network(&net, 1);
	int i;
	double start = this->get_time_point();		// Time in us
	image im = make_image(net.w, net.h, net.c);

	for (i = 0; i < trials; ++i) {
		network_predict(net, im.data);
	}

	double end = this->get_time_point();		// Time in us
	double took_time = std::chrono::duration<double>(end - start).count();

	return took_time / 1000.0;	// scale to ms from us
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



