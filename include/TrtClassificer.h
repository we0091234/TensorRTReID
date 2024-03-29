#ifndef TRTCLASSFIER_H_
#define TRTCLASSFIER_H_

// #include "logger.h"
#include "common.h"
// #include "argsParser.h"
// #include "buffers.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include<time.h>
#include "logging.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include<windows.h>
#include<string>
#include <iomanip>
#include <vector>
static sample::Logger gLogger;
using namespace nvinfer1;
class TrtClassificer
{
public:
	TrtClassificer(int INPUT_H,int INPUT_W,int CHANNELS,const char * INPUT_NAME,const char *OUTPUT_NAME,int outputSize);
	TrtClassificer(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME, std::vector<std::string>OUTPUT_NAME, int outputSize);
	void CaffeToGIEModel(const char* deployFile,const char* modelFile,const std::vector<std::string>& outputs, unsigned int maxBatchSize, const char * TrtSaveFileName);
	void doInference( float* input, float* output, int batchSize);
	void doInference(IExecutionContext& context, float* input, float **&output, int batchSize);
	void saveToTrtModel(const char * TrtSaveFileName)
	{
		std::ofstream out(TrtSaveFileName, std::ios::binary);
		if (!out.is_open())
		{
		std::cout << "���ļ�ʧ��!" <<std:: endl;
		}
		out.write(reinterpret_cast<const char*>(this->_gieModelStream->data()), this->_gieModelStream->size());
		out.close();
	}
	void  readTrtModel(const char * Trtmodel)
	{
		size_t size{ 0 };
		std::ifstream file(Trtmodel, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			_trtModelStream = new char[size];
			assert(_trtModelStream);
			file.read(_trtModelStream, size);
			file.close();
		}
		_trtModelStreamSize = size;

		_runtime = createInferRuntime(gLogger);
		_engine1 = _runtime->deserializeCudaEngine(_trtModelStream, _trtModelStreamSize);
		//cudaSetDevice(0);
		context = _engine1->createExecutionContext();
	}
	
	//void mydoInference(IExecutionContext& context, float* input, float *output, int batchSize);

	~TrtClassificer()
	{
		free(_inputName);
		free(_outputName);
		if (_gieModelStream)
			_gieModelStream->destroy();
		delete _trtModelStream;
		context->destroy();
		_engine1->destroy();
		_runtime->destroy();

	}
	int _input_h;
	int _input_w;
	int _channel;
	nvinfer1::IHostMemory *_gieModelStream{ nullptr };
	char *_inputName;
	char *_outputName;
	int  _outputNumber;
	char *_trtModelStream{ nullptr };
	int _trtModelStreamSize;
	IRuntime* _runtime;
	ICudaEngine* _engine1;
	IExecutionContext *context;
};
#endif // !1