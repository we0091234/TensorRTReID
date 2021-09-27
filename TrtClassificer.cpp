#include "TrtClassificer.h"
using namespace nvcaffeparser1;

//static sample::Logger gLogger;
TrtClassificer::TrtClassificer(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME, const char *OUTPUT_NAME, int outputSize)
{
	this->_input_h = INPUT_H;
	this->_input_w = INPUT_W;
	this->_channel = CHANNELS;
	this->_inputName = strdup(INPUT_NAME);
	this->_outputName = strdup(OUTPUT_NAME);
	this->_outputNumber = outputSize;

}

void TrtClassificer::CaffeToGIEModel(const char* deployFile, const char* modelFile, const std::vector<std::string>& outputs, unsigned int maxBatchSize, const char * TrtSaveFileName)
{

	std::cout << "Convert Caffemodel to  Trt  model...." << std:: endl;
	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetworkV2(0U);
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile, modelFile, *network, nvinfer1::DataType::kFLOAT);
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20); 
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	network->destroy();
	parser->destroy();
	this->_gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
	this->saveToTrtModel(TrtSaveFileName);
	std::cout << "Convert Done!" << std::endl;
}

void TrtClassificer::doInference(float* input, float* output, int batchSize)
{
	
	const ICudaEngine& engine= (*context).getEngine();
	assert(engine.getNbBindings() == 2);
	void* buffers[2];
	int inputIndex = engine.getBindingIndex(this->_inputName),
		outputIndex = engine.getBindingIndex(this->_outputName);
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * this->_channel * this->_input_h * this->_input_w * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * this->_outputNumber * sizeof(float)));
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * this->_channel * this->_input_h * this->_input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
	(*context).enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize *this->_outputNumber * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}