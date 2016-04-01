#include "classify.h" 
#include "classMethods.h"

int runTrainTestClassifier(const int classMethId, ClassData & tempDat);

#define USE_STANDARD 0	
#define USE_FUNCTION 0

int main()
{
	//
	// create a data set with two gaussian clusters
	//
	
	const int ndata = 200, ndim = 3;
	const float trainRatio = 0.75f; // fraction of data used for training
	int ntrain = (int)(trainRatio*ndata);
	int ntest  =  ndata - ntrain;

	std::cout  << trainRatio*ndata << " ntrain=" << ntrain << ", ntest=" << ntest << std::endl;

	Mat dset = Mat::zeros(ndata, ndim, CV_32FC1);
	randn(dset,0,10);
	Mat label = Mat::zeros(ndata,1,CV_32FC1);

	// create two Gauss Clusteres: shift one half and assign label
	int idat,idim;
	for(idat=ndata/2; idat<ndata;++idat)
	{
		label.at<float>(idat)=1;
		for(idim=0;idim<ndim;++idim)
		{
			dset.at<float>(idat,idim)+=15;
		}
	}

	// create index
	setIdxT idx;
	idx.train= Mat::zeros(ntrain,1,CV_32SC1) - 1;
	idx.test = Mat::zeros(ntest ,1,CV_32SC1) - 1;

	Classify::splitDataSet( &idx, ndata, trainRatio);

	// set up test set with labels
	Mat testSet = Mat::zeros(ntest,ndim,CV_32FC1);
	LABEL_TYPE testLabel_true(ntest,0);
	LABEL_TYPE testLabel_pred(ntest,0);

	for(idat=0;idat<ntest;++idat)
	{
		testSet.row(idat) = dset.row( idx.test.at<int>(idat) ) + 0; //has to be expression to work with .row!
		testLabel_true[idat] = label.at<float>( idx.test.at<int>(idat) );
	}
	
	//
	// store different classes in one pointer array
	//
 
	std::cout << "Using standard ..." << std::endl;
	const int nclassMeth = 5;
	Classify * classMeth = NULL; // declare pointer

	if (USE_STANDARD)
	{
	// initialize pointer to classifier method
	for(int icm=0;icm<=nclassMeth;icm++)
	{
		std::cout<<"case "<<icm<<std::endl;
		switch (icm)
		{		
			case 0:
				classMeth = new Classify;
				break;
			case 1:
				classMeth = new Bayes;
				break;
			case 2:
				classMeth = new KNN;
				break;
			case 3:
				classMeth = new RanTree;
				break;
			case 4:
				classMeth = new cSVMlin;
				break;
			case 5:
				classMeth = new cSVMrbf;
				break;

			default:
				std::cout<<"Not implemented"<<std::endl;
				break;
		}
		classMeth->findBestParam(dset,label, idx.train,testSet, testLabel_true, testLabel_pred);
		classMeth->train( dset, label, idx.train);
		
		testLabel_pred.assign(ntest,0);
		classMeth->test( testSet, testLabel_pred);
		std::cout << std::setw(70) << " Accuracy= " << classMeth->calcClassPerf(testLabel_true, testLabel_pred) << std::endl;

	    delete classMeth;// free memory	
	}
	
	// clean up
	~dset;
	~label;
	}	
	//
	// use the ClassData 
	//
	
	std::cout<<" test: "<<std::endl;
	ClassData tempDat;
	tempDat.showDataInfo();
	tempDat.setDataDim(200,3);
	tempDat.showDataInfo();

	std::cout<<" label     : "<< tempDat.getLabel(3)<< std::endl;	
	std::cout<<" data      : "<< tempDat.getData(0,2)<< std::endl;
	
	createRanDataSet(tempDat);
	
	tempDat.setTrainRatio(0.75f);
	std::cout<<"t-ratio= "<<tempDat.trainRatio<<" ntrain= "<<tempDat.ntrain
		<<" ntest= "<<tempDat.ntest<<std::endl;

	Classify::splitDataSet( &tempDat.idx, tempDat.nele, tempDat.trainRatio);
	
	// show training set
	//std::cout<<tempDat.idx.train.rows<<" "<<tempDat.idx.train.cols<< std::endl;
	int k=0; 
	//int tempIdx;
	//for(k=0;k<tempDat.ntrain;k++)
	//{
	//	//std::cout<<tempDat.idx.train.at<int>(k)<<" "<<tempDat.getData( tempDat.idx.train.at<int>(k) , 1) << std::endl;
	//	tempIdx = tempDat.idx.train.at<int>(k);
	//	std::cout<<tempIdx<<" "<<tempDat.dset.row( tempIdx) << " "
	//		     <<tempDat.getLabel(tempIdx) << std::endl;
	//}

	//create test set
	std::cout<<"test set      :"<< std::endl;
	tempDat.createTestSet();

	/*for(k=0;k<tempDat.ntest;k++)
	{
		std::cout<< tempDat.testSet.row(k) << std::endl;
	}*/	

	if (USE_FUNCTION)
	{
		std::cout<<"...using function      :"<< std::endl;

		for(int icm=0;icm<=nclassMeth;icm++)
		{
			runTrainTestClassifier(icm, tempDat);
		}
	}

	return 0;
}
//
//
//
int runTrainTestClassifier(const int classMethId, ClassData & tempDat)
{
	Classify * classMeth; // declare pointer to the classifier method

	// select the corresponding classifier class
	switch (classMethId)
	{
		std::cout<<"case "<<classMethId<<std::endl;
		case 0:
			classMeth = new Classify;
			break;
		case 1:
			classMeth = new Bayes;
			break;
		case 2:
			classMeth = new KNN;
			break;
		case 3:
			classMeth = new RanTree; 
			break;
		case 4:
			classMeth = new cSVMlin;
			break;
		case 5:
			classMeth = new cSVMrbf();
			break;

		default:
			std::cout<<"Invalid classifier name"<<std::endl;
			return -1; 
	}
				
	classMeth->findBestParam(tempDat.dset,tempDat.label, tempDat.idx.train,
								tempDat.testSet, tempDat.testLabel_true, tempDat.testLabel_pred);
	classMeth->train( tempDat.dset, tempDat.label, tempDat.idx.train );
		
	tempDat.testLabel_pred.assign(tempDat.ntest,0);
	classMeth->test(tempDat.testSet, tempDat.testLabel_pred );
	std::cout << std::setw(70) << " Accuracy= " << classMeth->calcClassPerf(tempDat.testLabel_true, tempDat.testLabel_pred) << std::endl;
		
	delete classMeth;// free memory
	std::cout << classMeth << std::endl;

	return 0;
}

//int runTrainTestClassifier(const int classMethId, ClassData & tempDat)
//{
//	//Classify classMeth; // declare pointer to the classifier method
//	cSVMrbf * classMeth = new cSVMrbf;
//	// select the corresponding classifier class
//				
//	classMeth->findBestParam(tempDat.dset,tempDat.label, tempDat.idx.train,
//								tempDat.testSet, tempDat.testLabel_true, tempDat.testLabel_pred);
//	classMeth->train( tempDat.dset, tempDat.label, tempDat.idx.train );
//		
//	tempDat.testLabel_pred.assign(tempDat.ntest,0);
//	classMeth->test(tempDat.testSet, tempDat.testLabel_pred );
//	std::cout << std::setw(70) << " Accuracy= " << classMeth->calcClassPerf(tempDat.testLabel_true, tempDat.testLabel_pred) << std::endl;
//	
//	
//	std::cout << classMeth <<" "<< &tempDat << std::endl;
//	
//	//CvNormalBayesClassifier * bla = new CvNormalBayesClassifier;
//	//bla->clear();
//    delete classMeth;// free memory
//	return 0;
//}
////classification
	//classRun.findBestParam(tempDat.dset,tempDat.label, tempDat.idx.train,
	//					   tempDat.testSet, tempDat.testLabel_true, tempDat.testLabel_pred);
	//classRun.train( tempDat.dset, tempDat.label, tempDat.idx.train);
	//
	//tempDat.testLabel_pred.assign(tempDat.ntest,0);
	//classRun.test( tempDat.testSet, tempDat.testLabel_pred);
	//std::cout << std::setw(70) << " Accuracy= " << classRun.calcClassPerf(tempDat.testLabel_true, tempDat.testLabel_pred) << std::endl;
	//
	//*for(k=0;k<tempDat.ntest;k++)
	//{
	//	std::cout<< tempDat.testLabel_true[k] <<" "<< tempDat.testLabel_pred[k]  << std::endl;
	//}*/