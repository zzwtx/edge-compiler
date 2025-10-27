#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"


// 记录 TensorRT 运行时的日志信息
class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* msg)  noexcept
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

// 加载标签文件获得分类标签
std::string labels_txt_file = "D:/C++_demo_tensort/tensorrt/flower_classes.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}

int main(int argc, char** argv) {
	// 预测的目标标签数
	std::vector<std::string> labels = readClassNames();

	// engine训练模型文件
	std::string enginepath = "D:/C++_demo_tensort/tensorrt/AlexNet.engine";

	// 从文件中读取一个序列化的 TensorRT 模型
	std::ifstream file(enginepath, std::ios::binary);
	char* trtModelStream = NULL;
	int size = 0;
	if (file.good()) {
		// 将读指针移动到文件末尾
		file.seekg(0, file.end);
		// 获取文件大小
		size = file.tellg();
		// 将读指针移动到文件开始
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		// 从关联的输入流中读取了指定数量的字符
		file.read(trtModelStream, size);
		file.close();
	}

	// 初始化推理运行时对象
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	assert(runtime != nullptr);

	// 加载预先构建好的引擎
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(this->engine != nullptr);

	// 创建执行上下文
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	assert(this->context != nullptr);

	// 释放数组类型的内存
	delete[] trtModelStream;

	// 管理异步操作的流对象
	cudaStream_t stream;

	// 对象的绑定数量(即输入和输出的总数)
	int num_bindings = engine->getNbBindings();
	std::cout << " input/outpu : " << num_bindings << std::endl;
	// 输入和输出名称
	std::vector<const char*> input_names;
	std::vector<const char*> output_names;
	// 遍历所有绑定
	for (int i = 0; i < num_bindings; ++i) {
		// 获取绑定名称
		const char* binding_name = engine->getBindingName(i);

		// 判断当前绑定是输入还是输出,并保存到相应的向量中
		if (engine->bindingIsInput(i)) {
			input_names.push_back(binding_name);
		}
		else {
			output_names.push_back(binding_name);
		}
	}
	// 用于获取模型输入或输出张量的索引
	int input_index = engine->getBindingIndex(input_names[0]);
	int output_index = engine->getBindingIndex(output_names[0]);

	// 获取输入维度信息 NCHW
	int input_h = engine->getBindingDimensions(input_index).d[2];
	int input_w = engine->getBindingDimensions(input_index).d[3];
	printf("inputH : %d, inputW: %d \n", input_h, input_w);

	// 获取输出维度信息
	int output_h = engine->getBindingDimensions(output_index).d[0];
	int output_w = engine->getBindingDimensions(output_index).d[1];
	printf("outputH : %d, outputW: %d \n", output_h, output_w);

	// 推理准备
	// 包含所有输入和输出缓冲区的指针,每个元素对应一个输入或输出绑定,顺序与模型中绑定的顺序一致
	void* buffers[2] = { NULL, NULL };
	// 创建GPU显存输入/输出缓冲区(有几个就初始化几个)
	cudaMalloc(&buffers[input_index], input_h*input_w*3*sizeof(float));
	cudaMalloc(&buffers[output_index], output_h*output_w*sizeof(float));

	// 输出结果
	std::vector<float> prob;
	// 创建临时缓存输出
	prob.resize(output_h*output_w);
	// 创建cuda流
	cudaStreamCreate(&stream);
	
	// 测试图片
	cv::Mat image = cv::imread("D:/C++_demo_tensort/tensorrt/sunflowers.jpg");

	// 预处理输入数据
	cv::Mat rgb, blob;
	// 默认是BGR需要转化成RGB
	cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
	// 对图像尺寸进行缩放
	cv::resize(rgb, blob, cv::Size(input_w, input_h));
	blob.convertTo(blob, CV_32F);
	// 对图像进行标准化处理
	blob = blob / 255.0;
	cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
	cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);
	// CHW-->NCHW 维度扩展
	cv::Mat tensor = cv::dnn::blobFromImage(blob);

	// 内存到GPU显存
	cudaMemcpyAsync(buffers[0], tensor.ptr<float>(), input_h*input_w*3*sizeof(float), cudaMemcpyHostToDevice, stream);
	// 模型推理
	context->enqueueV2(buffers, stream, nullptr);
	// GPU显存到内存
	cudaMemcpyAsync(prob.data(), buffers[1], output_h*output_w*sizeof(float), cudaMemcpyDeviceToHost, stream);

	// 后处理推理结果
	cv::Mat probmat(output_h, output_w, CV_32F, (float*)prob.data());
	cv::Point maxL, minL;		// 用于存储图像分类中的得分最小值索引和最大值索引(坐标)
	double maxv, minv;			// 用于存储图像分类中的得分最小值和最大值
	cv::minMaxLoc(probmat, &minv, &maxv, &minL, &maxL);
	int max_index = maxL.x;		// 获得最大值的索引,只有一行所以列坐标既为索引
	std::cout << "label id: " << max_index << std::endl;
	// 在测试图像上加上预测的分类标签
	cv::putText(image, labels[max_index], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
	cv::imshow("输入图像", image);
	cv::waitKey(0);

	// 同步结束，释放资源
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);

	if (!context) {
		context->destroy();
	}
	if (!engine) {
		engine->destroy();
	}
	if (!runtime) {
		runtime->destroy();
	}
	if (!buffers[0]) {
		delete[] buffers;
	}

	return 0;
}
