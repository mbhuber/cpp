#ifndef _CLASSIFY_H_
#define _CLASSIFY_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <valarray>
#include <iomanip>

using namespace cv;

// types
#define LABEL_TYPE std::vector<float> // improve: two labels types are used

struct setIdxT
{
	//setIdxT();
	//setIdxT(Mat _train, Mat _test);
	
	Mat train;
	Mat  test;	
};

const string classMeth[] = {"BAYES","kNN","SVMlin","SVMrbf","nuSVMlin"};
//const enum classMeth_t {BAYES=0, KNN=1, SVM_LIN=2, SVM_RBF=3, NU_SVM_LIN=4};

// base class for all classifier methods

class Classify
{
public:
	Classify();
	//Classify(Mat & dataSet, Mat & labels, Mat & trainSetIndex);

	virtual ~Classify();

	virtual void train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);

	virtual void test(const Mat & testSet, std::vector<float> & predLabels); 

	int findBestParam(const Mat & dataSet, const Mat & labels, Mat & trainSetIndex,
							 const Mat & testSet, const std::vector<float> & trueLabels, std::vector<float> & predLabels);

	float CrossValid(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);
	
	static void splitDataSet(setIdxT * idx,const int ndata, const float trainRatio);
	
	int maxPerf(int * maxIdx, float * accu, int r, int c);

	float calcClassPerf(const std::vector<float> & trueLabel,const std::vector<float> & predLabel);

protected:

	//static const int npara=6; // elements for one parameter that are tried

	//CvStatModel * model; // pointer for all classfier models !!! does not work
	
	struct freePara_t
	{
		int npara;
		float all[10];
		float use;
	};

	struct modelParam_t
	{
		int nFreeParam;
		freePara_t freePara_1;
		freePara_t freePara_2;
	};

	modelParam_t * modelParam;
};

class ClassData 
{
public:
	ClassData();
	virtual ~ClassData(){};

	void setDataDim( int nele, int dims ); 
	void showDataInfo();
	// data input/output
	float getLabel(int idx);
	float getData(int idx, int idim);
	void setLabel(int idx, float value);
	void setData(int idx, int idim, float value);
	// setting training/test set
	void setTrainRatio(float value);

	void createTestSet();
	//void loadData();
	//Mat & getData();
	int nele;
	int ndim;
	
	Mat label;
	Mat dset;

	float trainRatio;
	int ntrain;
	int ntest;
	setIdxT idx; // split index

	Mat testSet;
	LABEL_TYPE testLabel_true;
	LABEL_TYPE testLabel_pred;

private:
	void updateData();

};

// auxiliary function

void createRanDataSet(ClassData & dset);

#endif