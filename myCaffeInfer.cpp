#include "TrtClassificer.h"
#include "utils.h"
#include <iosfwd>
#include "scanfFile.h"
#include "dlib/clustering/chinese_whispers.h"
#include <fstream>
#define  numAttribute 11
#define shortSSE  0
#define fileSize 100
#define IdSize 50

using namespace nvcaffeparser1;
using namespace cv;
using namespace std;

static const int INPUT_H = 144; //输入图像高
static const int INPUT_W = 96;//输入图像宽
static const int CHANNELS = 3;//输入图像通道
const int OUTPUT_SIZE = 512;//输出特征维度
const char* INPUT_BLOB_NAME = "data";//deploy文件中定义的输入层名称
const char* OUTPUT_BLOB_NAME = "batch_norm_blob2";//deploy文件中定义的输出层名称

typedef struct idSimilarity
{
	char  ID[50];
	float similarity;
}IdwithSimi;
typedef struct Query
{
	char fileName[100];
	char PerID[10];
	vector<pair<string,float>> gallerySimilarity;
	vector<float >fe;
	short feShort[OUTPUT_SIZE];
}myQuery;
class imageFe
{
public:
	imageFe()
	{
		fileName = new char[fileSize];
		PerID = new char[IdSize];
		fe = new float[OUTPUT_SIZE];
		feShort = new short[OUTPUT_SIZE];
	}
	void _init(int size)
	{
		idWithSimi = new IdwithSimi[size];
	}
	
	~imageFe()
	{
		if (idWithSimi)
			delete[] idWithSimi;
		if (fileName)
			delete[] fileName;
		if (PerID)
			delete[] PerID;
		if (fe)
			delete[] fe;
		if (feShort)
			delete[] feShort;
	}


	IdwithSimi * idWithSimi = NULL;
	char *fileName = NULL;
	char *PerID = NULL;
	float *fe = NULL;
	short *feShort = NULL;

};
bool judge_1(IdwithSimi a, IdwithSimi b)
{
	return a.similarity > b.similarity;
}
void getFeature(TrtClassificer  &myClassifier,vector<string>&querryFilelist,int i,float *&data,float *mean_data, float &sumTime, vector<myQuery>&queryData)
{
	myQuery  tempQuery;
	string PersonId = getPersonId(querryFilelist[i]);
	strcpy(tempQuery.fileName, querryFilelist[i].c_str());
	strcpy(tempQuery.PerID, PersonId.c_str());
	float *feature = new float[OUTPUT_SIZE];
	string fileName = querryFilelist[i];
	imageProcess((char *)fileName.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);
	float *prob = new float[OUTPUT_SIZE];
	clock_t starts1 = clock();
	myClassifier.doInference(data, feature, 1);
	clock_t ends1 = clock();
	sumTime += ends1 - starts1;
	if (i % 1000 == 0 && i != 0)
	{
		std::cout << "Query入库：" << i << "张" << "每张耗时:" << ends1 - starts1 << std::endl;
	}
	normalizex(OUTPUT_SIZE, feature);
	//std::cout << ends1 - starts1 << std::endl;
	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		tempQuery.fe.push_back(feature[i]);
	}
	/*normalizex(OUTPUT_SIZE, tempQuery.fe);*/

	if (shortSSE)
	{
		float scale = (int)0x7fff;
		int *vali1 = new int[OUTPUT_SIZE];
		//short *valshort1 = new short[OUTPUT_SIZE];
		for (int i = 0; i < OUTPUT_SIZE; i++)
		{
			vali1[i] = min(scale, tempQuery.fe[i] * scale + 0.5f);
			tempQuery.feShort[i] = vali1[i];
		}
		delete[] vali1;
	}

	queryData.push_back(tempQuery);
	delete[] feature;
	delete[] prob;
	 
}
struct reidstruct
{
	string name;
	vector<short>feat_short;
	vector<float>feat_float;
};

int main(/*int argc,char **argv*/)
{
	cudaSetDevice(0);

	// //clock_t sumtimeBegin = clock();
	double sumtimeBegin=static_cast<double>(getTickCount());

	// int sumLabel = 0;
	// int rightLabel = 0;
	// int errorLabel = 0;
	std::vector<dlib::sample_pair> edges;
	float scale = (int)0x7fff;
	// const char *gender[] = { "女","男" };
	// IHostMemory *gieModelStream{ nullptr };
	const char *modelFile = "../reuslt/net_76.caffemodel";
	const char *deployFile = "../reuslt/net_76.prototxt";
	// char *filePath = "E:\\REID\\Reid\\1";
	char *trtSavePath = "/home/cxl/tensorCaffe/caffeTensorRT/ReID76.trt";
	char * queryPath = "/home/cxl/data/Market/query";
	char * galleryPath = "/home/cxl/data/Market/gallery";
	float *data = new float[INPUT_H*INPUT_W*CHANNELS];
	float mean_data[] = { 97.59758647 , 99.04790283, 104.8204798 };
	vector<string>querryFilelist;
	querryFilelist.clear();
	vector<string>GalleryFileList;
	GalleryFileList.clear();
	vector<int>label;
	TrtClassificer myClassifier(INPUT_H, INPUT_W, CHANNELS, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, OUTPUT_SIZE);
	myClassifier.CaffeToGIEModel(deployFile, modelFile, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, trtSavePath);
	myClassifier.readTrtModel(trtSavePath);
	// size_t size{ 0 };
	vector<string> fileType={"jpg"};
	// readDir(queryPath, querryFilelist, label);
	readFileList(queryPath,querryFilelist,fileType);
	readFileList(galleryPath,GalleryFileList,fileType);
	// readDir(galleryPath, GalleryFileList, label);

	double sumTime = 0;
	imageFe *queryData1 = new imageFe[querryFilelist.size()];
	for (int i = 0; i < querryFilelist.size(); i++)
	{
		queryData1[i]._init(GalleryFileList.size());
	}
	imageFe *galleryData1 = new imageFe[GalleryFileList.size()];
	//myQuery  tempQuery;

	for (int i = 0; i < querryFilelist.size(); i++)
	{
		
		string PersonId = getPersonId(querryFilelist[i]);
		strcpy(queryData1[i].fileName, querryFilelist[i].c_str());
		strcpy(queryData1[i].PerID, PersonId.c_str());
		float *feature = new float[OUTPUT_SIZE];
		string fileName = querryFilelist[i];
		imageProcess((char *)fileName.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);
		float *prob = new float[OUTPUT_SIZE];
		//clock_t starts1 = clock();
		double starts1=static_cast<double>(getTickCount());
		myClassifier.doInference(data, feature, 1);
		//clock_t ends1 = clock();
		double ends1=static_cast<double>(getTickCount());
		sumTime += ends1 - starts1;
		if (i % 1000 == 0 && i != 0)
		{
			std::cout << "Query: " << i << " pics" << " time:" << (ends1 - starts1)/getTickFrequency()*1000 << std::endl;
		}
		normalizex(OUTPUT_SIZE, feature);
		//std::cout << ends1 - starts1 << std::endl;
		for (int k = 0; k < OUTPUT_SIZE; k++)
		{
			queryData1[i].fe[k]=feature[k];
		}
		/*normalizex(OUTPUT_SIZE, tempQuery.fe);*/

		if (shortSSE)
		{
			float scale = (int)0x7fff;
			int *vali1 = new int[OUTPUT_SIZE];
			//short *valshort1 = new short[OUTPUT_SIZE];
			for (int i1 = 0; i1 < OUTPUT_SIZE; i1++)
			{
				vali1[i1] = min(scale, queryData1[i].fe[i1] * scale + 0.5f);
				queryData1[i].feShort[i1] = vali1[i1];
				//tempQuery.feShort[i] = vali1[i];
			}
			delete[] vali1;
		}

		
		delete[] feature;
		delete[] prob;
	}

	for (int i = 0; i < GalleryFileList.size(); i++)
	{

		string PersonId = getPersonId(GalleryFileList[i]);
		strcpy(galleryData1[i].fileName, GalleryFileList[i].c_str());
		strcpy(galleryData1[i].PerID, PersonId.c_str());
		float *feature = new float[OUTPUT_SIZE];
		string fileName = GalleryFileList[i];
		imageProcess((char *)fileName.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);
		float *prob = new float[OUTPUT_SIZE];
		//clock_t starts1 = clock();
		double starts1=static_cast<double>(getTickCount());
		myClassifier.doInference(data, feature, 1);
		//clock_t ends1 = clock();
		double ends1=static_cast<double>(getTickCount());
		sumTime += ends1 - starts1;
		if (i % 1000 == 0 && i != 0)
		{
			std::cout << "Gallery: " << i << " pics" << " time:" << (ends1 - starts1)/getTickFrequency()*1000 << std::endl;
		}
		normalizex(OUTPUT_SIZE, feature);
		//std::cout << ends1 - starts1 << std::endl;
		for (int k = 0; k < OUTPUT_SIZE; k++)
		{
			galleryData1[i].fe[k] = feature[k];
		}
		/*normalizex(OUTPUT_SIZE, tempQuery.fe);*/


		if (shortSSE)
		{
			float scale = (int)0x7fff;
			int *vali1 = new int[OUTPUT_SIZE];
			//short *valshort1 = new short[OUTPUT_SIZE];
			for (int i1 = 0; i1 < OUTPUT_SIZE; i1++)
			{
				vali1[i1] = min(scale, galleryData1[i].fe[i1] * scale + 0.5f);
				galleryData1[i].feShort[i1] = vali1[i1];
				//tempQuery.feShort[i] = vali1[i];
			}
			delete[] vali1;
		}


		delete[] feature;
		delete[] prob;
	}



	float mAP = 0;
	// float sumTime1 = 0;
	for (int i = 0; i <  querryFilelist.size(); i++)
	{
		int t = 0;
		float rank = 0;
		float rank_ap = 0;
		//clock_t starts2 = clock();
		double starts2=static_cast<double>(getTickCount());
		for (int j = 0; j <GalleryFileList.size(); j++)
		{
			float similary;
			if (shortSSE)
			{
				int sum1 = dot_short_SSE(OUTPUT_SIZE, queryData1[i].feShort, galleryData1[j].feShort);
				similary = (1.0*sum1) / (scale*scale);
			}
			else
			{

				similary = getSimilarity1(queryData1[i].fe, galleryData1[j].fe, OUTPUT_SIZE);
			}
				strcpy(queryData1[i].idWithSimi[j].ID, galleryData1[j].PerID);
				queryData1[i].idWithSimi[j].similarity = similary;
			
		}
		sort(queryData1[i].idWithSimi, queryData1[i].idWithSimi+ GalleryFileList.size(), judge_1);

		for (int k = 0; k < GalleryFileList.size(); k++)
		{
			if (strcmp(queryData1[i].PerID, queryData1[i].idWithSimi[k].ID) == 0)
			{
				t++;
				rank = rank + (float)t / (k + 1);
			}
		}
		if (t == 0)
		{

			std::cout << "---error---gallery has no query pic!" << endl;
			rank_ap = 0;
		}
		else
		{
			rank_ap = (float)rank / t;
		}
		mAP = mAP + rank_ap;
		//clock_t ends2 = clock();
		double ends2=static_cast<double>(getTickCount());
		std::cout << "QuerrycomparDown " << i << " pics" << " time:" << (ends2 - starts2)/getTickFrequency()*1000<< "ms" << std::endl;
	}
	delete[] data;
	std::cout << "Map: " << mAP / querryFilelist.size() << std::endl;
	//clock_t sumTimeEnds = clock();
	double sumTimeEnds=static_cast<double>(getTickCount());
	double secondTime= (sumTimeEnds - sumtimeBegin )/getTickFrequency();
	std::cout << "allTime: " << secondTime << " s " << secondTime/60 << " minute" << std::endl;
	delete[] queryData1;
	delete[] galleryData1;
	// system("pause");

}


int main_julei(/*int argc,char **argv*/)
{
	cudaSetDevice(0);

	// //clock_t sumtimeBegin = clock();
	double sumtimeBegin=static_cast<double>(getTickCount());

	// int sumLabel = 0;
	// int rightLabel = 0;
	// int errorLabel = 0;
	
	float scale = (int)0x7fff;
	// const char *gender[] = { "女","男" };
	// IHostMemory *gieModelStream{ nullptr };
	const char *modelFile = "/home/cxl/tensorCaffe/caffeTensorRT/model/net_36.caffemodel";
	const char *deployFile = "/home/cxl/tensorCaffe/caffeTensorRT/model/net_36.prototxt";
	// char *filePath = "E:\\REID\\Reid\\1";
	char *trtSavePath = "/home/cxl/tensorCaffe/caffeTensorRT/ReID.trt";
    float threshold = 0.9;
	char * galleryPath = "/home/data/cxl/ReidTest/oriPic";
	string txtname = "rm_repeatID_same_id_" +to_string(threshold)+".txt";
	float *data = new float[INPUT_H*INPUT_W*CHANNELS];
	float mean_data[] = { 97.59758647 , 99.04790283, 104.8204798 };

	vector<string>GalleryFileList;
	GalleryFileList.clear();
	vector<int>label;
	TrtClassificer myClassifier(INPUT_H, INPUT_W, CHANNELS, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, OUTPUT_SIZE);
	// myClassifier.CaffeToGIEModel(deployFile, modelFile, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, trtSavePath);
	myClassifier.readTrtModel(trtSavePath);
	// size_t size{ 0 };
	vector<string> fileType={"jpg"};
	// readDir(queryPath, querryFilelist, label);

	readFileList(galleryPath,GalleryFileList,fileType);
	// readDir(galleryPath, GalleryFileList, label);

	double sumTime = 0;
	
	//myQuery  tempQuery;
    
    vector<reidstruct> baseFeat;
	cout<<"extractFeatureStart!"<<endl;
	for (int i = 0; i < GalleryFileList.size(); i++)
	{

		string PersonId = getPersonId(GalleryFileList[i]);
		float *feature = new float[OUTPUT_SIZE];
		string fileName = GalleryFileList[i];
		imageProcess((char *)fileName.c_str(), data, mean_data, CHANNELS, INPUT_H, INPUT_W);
		//clock_t starts1 = clock();
		double starts1=static_cast<double>(getTickCount());
		myClassifier.doInference(data, feature, 1);
		//clock_t ends1 = clock();
		double ends1=static_cast<double>(getTickCount());
		sumTime += ends1 - starts1;
        reidstruct tep;
		tep.name=GalleryFileList[i];
		
		if (i % 10 == 0 && i != 0)
		{
			std::cout << "Gallery: " << i << " pics" << " time:" << (ends1 - starts1)/getTickFrequency()*1000 << std::endl;
		}
		normalizex(OUTPUT_SIZE, feature);
		//std::cout << ends1 - starts1 << std::endl;
		for (int k = 0; k < OUTPUT_SIZE; k++)
		{
			// galleryData1[i].fe[k] = feature[k];
			tep.feat_float.push_back(feature[k]);
		}
		/*normalizex(OUTPUT_SIZE, tempQuery.fe);*/


		if (shortSSE)
		{
			float scale = (int)0x7fff;
			int *vali1 = new int[OUTPUT_SIZE];
			//short *valshort1 = new short[OUTPUT_SIZE];
			for (int i1 = 0; i1 < OUTPUT_SIZE; i1++)
			{
				vali1[i1] = min(scale, tep.feat_float[i1] * scale + 0.5f);
				tep.feat_short.push_back(vali1[i1]);
				//tempQuery.feShort[i] = vali1[i];
			}
			baseFeat.push_back(tep);
			delete[] vali1;
		}


		delete[] feature;
	}
   std::cout<<"extractFeatureCompleted!"<<std::endl;
   	std::vector<dlib::sample_pair> edges;
    for (int m = 0; m < baseFeat.size(); m++)
	{
		for (int n = m + 1; n < baseFeat.size(); n++)
		{
			//short f1[512] = { 0 };
			short *f1 = new short[OUTPUT_SIZE]();
			for (int t = 0; t < OUTPUT_SIZE;t++)
			{
				f1[t] = baseFeat[m].feat_short[t];
			}
			//short f2[512] = { 0 };
			short *f2 = new short[OUTPUT_SIZE]();
			for (int t = 0; t < OUTPUT_SIZE; t++)
			{
				f2[t] = baseFeat[n].feat_short[t];
			}
			int sum;
			float similary;
			if (shortSSE)
			{
				int sum1 = dot_short_SSE(OUTPUT_SIZE, f1, f2);
				similary = (1.0*sum1) / (scale*scale);
			}
			else
			similary = getSimilarity(baseFeat[m].feat_float, baseFeat[n].feat_float, OUTPUT_SIZE);
			
			//cout << similar << endl;
			if (similary < threshold)
			{
				delete[] f1;
				delete[] f2;
				continue;
			}
			dlib::sample_pair *tep = new dlib::sample_pair(m, n, similary);
			edges.push_back(*tep);
			delete[] f1;
			delete[] f2;
		}
	}


	std::vector<unsigned long> labels;
	unsigned long num_iterations = 50;
	dlib::rand rnd;
	const auto num_clusters = dlib::chinese_whispers(edges, labels);	//, num_iterations, rnd
	printf("clustering nums:%d\n", num_clusters);
	vector<vector<int>>class_index;
	// 保存分类的图片
	for (int c = 0; c < num_clusters; c++)
	{
		vector<int> temp;
		for (int ind = 0; ind < labels.size(); ind++)
		{
			if (labels[ind] == c)
			{
				temp.push_back(ind);
			}
		}
		class_index.push_back(temp);
	}
    
	ofstream fout(txtname);
	for (int i = 0; i < class_index.size(); i++)
	{
		for (int j = 0; j < class_index[i].size(); j++)
		{
			fout << baseFeat[class_index[i][j]].name << " ";
		}
		fout << endl;
	}
	fout.close();
	printf("clustering Done\n");
	std::system("pause");
	return 0;

	
	// system("pause");

}

