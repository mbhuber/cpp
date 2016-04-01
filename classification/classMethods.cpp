#include "classMethods.h"

Bayes::Bayes()
{
	std::cout << "Initializing Bayes..." << std::endl;
	modelParam->nFreeParam=0;
	bayes   = new CvNormalBayesClassifier;
}

void Bayes::train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	std::cout << "Train Bayes..." << std::endl;
	bayes->train(dataSet, labels, Mat(), trainSetIndex, false);
}

void Bayes::test(const Mat & testSet, LABEL_TYPE & predLabels)
{
	std::cout << "Test Bayes..." << std::endl;
	for(int idat=0; idat<testSet.rows; ++idat)
			predLabels[idat] = bayes->predict(testSet.row(idat));
}
//
// KNN
//
KNN::KNN()
{
	std::cout << "Initializing kNN..." << std::endl;

	modelParam->nFreeParam=1;

	modelParam->freePara_1.npara = 6;
	for(int i=0; i<modelParam->freePara_1.npara;++i)
		modelParam->freePara_1.all[i]= 2.0f*(i+1)+1;
	modelParam->freePara_1.use = modelParam->freePara_1.all[0]; //default value

	knn   = new CvKNearest;
}

void KNN::train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	//std::cout << "Train kNN..." << std::endl;
	knn->train(dataSet, labels, trainSetIndex, false, (int) modelParam->freePara_1.use);
}

void KNN::test(const Mat & testSet, LABEL_TYPE & predLabels)
{
	//std::cout << "Test kNN..." << modelParam->freePara_1.use << std::endl;
	for(int idat=0; idat<testSet.rows; ++idat)
			predLabels[idat] = knn->find_nearest(testSet.row(idat), (int)modelParam->freePara_1.use);
}
//
// c-SVMlin
//
cSVMlin::cSVMlin()
{
	std::cout << "Initializing cSVMlin..." << std::endl;

	// 1 free Parameter: cost
	modelParam->nFreeParam=1;
	modelParam->freePara_1.npara = 6;
	for(int i=0; i<modelParam->freePara_1.npara; ++i)
		modelParam->freePara_1.all[i]= (float) 0.001 * (float) pow(10.0,i);
	modelParam->freePara_1.use = modelParam->freePara_1.all[0]; //default value

	c_svm_lin   = new CvSVM;
}

void cSVMlin::train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	//std::cout << "Train cSVMtrain..." << modelParam->freePara_1.use  << std::endl;
	// Set up SVM’s parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-10);

	//std::cout << "cSVMtrain: default cost..." << params.C  << std::endl;
	params.C = modelParam->freePara_1.use;
	// Train the SVM
	c_svm_lin->train(dataSet, labels, Mat(), trainSetIndex, params);		
}

void cSVMlin::test(const Mat & testSet, LABEL_TYPE & predLabels)
{
	//std::cout << "Test cSVMlin..." << modelParam->freePara_1.use << std::endl;
	for(int idat=0; idat<testSet.rows; ++idat)
			predLabels[idat] = c_svm_lin->predict(testSet.row(idat));
}
//
// c-SVMrbf
//

cSVMrbf::cSVMrbf()
{
	std::cout << "Initializing cSVMrbf..." << std::endl;

	// 2 free Parameter: cost & gamma
	modelParam->nFreeParam=2;
	modelParam->freePara_1.npara = 6;
	modelParam->freePara_2.npara = 6;

	// set up cost parameter
	int i=0;
	for(i=0; i<modelParam->freePara_1.npara; ++i)
		modelParam->freePara_1.all[i]= (float) 0.0001 * (float) pow(10.0,i);
	modelParam->freePara_1.use = modelParam->freePara_1.all[0]; //default value

	// set up gamma parameter
	for(i=0; i<modelParam->freePara_2.npara; ++i)
		modelParam->freePara_2.all[i]= (float) 1.0e-5 * (float) pow(5.0,i);
	modelParam->freePara_2.use = modelParam->freePara_2.all[0]; //default value

	//c_svm_rbf   = new CvSVM;
	model  = new CvSVM;
}

//cSVMrbf::~cSVMrbf()
//{
//	std::cout << "Deleting cSVMrbf..." << model << std::endl;
//	model->clear();
//}

void cSVMrbf::train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	//std::cout << "Train cSVMtrain..." << modelParam->freePara_1.use <<" "<<modelParam->freePara_2.use<< std::endl;
	// Set up SVM’s parameters
	CvSVMParams params;

	// const parameters
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-10);

	// parameters that are trained/optimized
	params.C     = modelParam->freePara_1.use;
	params.gamma = modelParam->freePara_2.use; 
	// Train the SVM
	model->train(dataSet, labels, Mat(), trainSetIndex, params);		
}

void cSVMrbf::test(const Mat & testSet, LABEL_TYPE & predLabels)
{
	//std::cout << "Test cSVMrbf..." << modelParam->freePara_1.use <<" "<<modelParam->freePara_2.use  <<std::endl;
	for(int idat=0; idat<testSet.rows; ++idat)
			predLabels[idat] = model->predict(testSet.row(idat));
}

//
// RanTree : Random Tree
//

RanTree::RanTree()
{
	std::cout << "Initializing RanTree..." << std::endl;

	// 1 free Parameter: max. depth
	modelParam->nFreeParam=1;
	modelParam->freePara_1.npara = 6;
	for(int i=0; i<modelParam->freePara_1.npara; ++i)
		modelParam->freePara_1.all[i] = 2*(float(i)+1)+1;
	modelParam->freePara_1.use = modelParam->freePara_1.all[0]; //default value
	
	ran_tree   = new CvRTrees;
}

void RanTree::train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	//std::cout << "Train cSVMtrain..." << modelParam->freePara_1.use  << std::endl;
	// Set up SVM’s parameters
	CvRTParams params = CvRTParams();
	
	params.max_depth = (int) modelParam->freePara_1.use;

	ran_tree->train( dataSet, CV_ROW_SAMPLE, labels, Mat(),Mat(),Mat(),Mat(), params);		
}

void RanTree::test(const Mat & testSet, LABEL_TYPE & predLabels)
{
	//std::cout << "Test cSVMlin..." << modelParam->freePara_1.use << std::endl;
	for(int idat=0; idat<testSet.rows; ++idat)
			predLabels[idat] = ran_tree->predict(testSet.row(idat));
}