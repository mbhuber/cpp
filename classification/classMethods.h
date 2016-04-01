#ifndef _CLASSMETHODS_H_
#define _CLASSMETHODS_H_

#include "classify.h"

class Bayes : public Classify
{
public:
	Bayes();
	//Classify(Mat & dataSet, Mat & labels, Mat & trainSetIndex);

	virtual ~Bayes(){bayes->clear();};

	virtual void train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);

	virtual void test(const Mat & testSet, std::vector<float> & predLabels); 

protected:
	CvNormalBayesClassifier * bayes;
};


class KNN : public Classify
{
public:
	KNN();

	virtual ~KNN(){knn->clear();};

	virtual void train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);

	virtual void test(const Mat & testSet, std::vector<float> & predLabels); 

protected:
	CvKNearest * knn;
};


class cSVMlin : public Classify
{
public:
	cSVMlin();

	virtual ~cSVMlin(){ c_svm_lin->clear(); };

	virtual void train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);

	virtual void test(const Mat & testSet, std::vector<float> & predLabels); 

protected:
	CvSVM * c_svm_lin;
};


class cSVMrbf : public Classify
{
public:
	cSVMrbf();

	virtual ~cSVMrbf(){ model->clear(); };

	virtual void train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);

	virtual void test(const Mat & testSet, std::vector<float> & predLabels); 

protected:
	CvSVM * model;
};

class RanTree : public Classify
{
public:
	RanTree();

	virtual ~RanTree(){ ran_tree->clear(); };

	virtual void train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex);

	virtual void test(const Mat & testSet, std::vector<float> & predLabels); 

protected:
	CvRTrees * ran_tree;
};
#endif
