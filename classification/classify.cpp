#include "classify.h"

ClassData::ClassData()
{
	nele=-1;
	ndim=-2;
}

void ClassData::setDataDim(int ele, int dims)
{
	nele=ele;
	ndim=dims;
	ClassData::updateData();
}

void ClassData::showDataInfo()
{
	std::cout<<" nele= "<< nele << std::endl;
	std::cout<<" ndim= "<< ndim << std::endl;
}

void ClassData::updateData()
{
	dset  = Mat::zeros(nele, ndim, CV_32FC1);
	label = Mat::zeros(nele,1,CV_32FC1);
}

float ClassData::getLabel(int idx)
{
	float temp;
	if(idx<nele)
		temp = label.at<float>(idx);
	else
		temp = -1;

	return temp;
}

float ClassData::getData(int idx, int idim)
{
	float temp;
	if(idx<nele && idim<ndim)
		temp = dset.at<float>(idx,idim);
	else
		temp = -1;

	return temp;
}

void ClassData::setData(int idx, int idim, float value)
{
	if(idx<nele && idim<ndim)
		dset.at<float>(idx,idim)= value;
	else
		std::cerr << "index miss-match" << std::endl;
}

void ClassData::setLabel(int idx, float value)
{
	if(idx<nele)
		label.at<float>(idx)= value;
	else
		std::cerr << "index miss-match" << std::endl;
}

void ClassData::setTrainRatio(float value)
{
	if(value>0 && value < 1)
	{
		trainRatio = value;
		ntrain = (int)(trainRatio*nele);
		ntest  =  nele - ntrain;
		// declare split idx fields
		idx.train= Mat::zeros(ntrain,1,CV_32SC1) - 1;
		idx.test = Mat::zeros(ntest ,1,CV_32SC1) - 1;
	}
	else
		std::cerr << "train ration has bad value: " << value <<std::endl;
}

void ClassData::createTestSet()
{
	if(ntest>0)
	{
		testSet = Mat::zeros(ntest,ndim, CV_32FC1);
		testLabel_true.assign(ntest,-1);
		testLabel_pred.assign(ntest,-1);

		for(int k=0;k<ntest;++k)
		{
			testSet.row(k) = dset.row( idx.test.at<int>(k) ) + 0; //has to be expression to work with .row!
			testLabel_true[k] = label.at<float>( idx.test.at<int>(k) );
		}
	} else
	{
		std::cerr<<"test set index is not defined"<< std::endl;
	}
}
//
//
// Classify
//
//

Classify::Classify()
{
	modelParam= new modelParam_t;
	modelParam->nFreeParam=0;
	modelParam->freePara_1.npara=0;
	modelParam->freePara_2.npara=0;
	//std::cout << "This is the base class. No classifier method is included." << std::endl;
	//model  = NULL;
}

Classify::~Classify()
{
	//std::cout << "Calling Classify destructor..." << std::endl;
	//if (model!=NULL)
	//	model->clear();
	delete modelParam;
}

// empty method: is defined by derived class
void Classify::train(const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	std::cout << "No classifier included for training..." << std::endl;
}

// empty method: is defined by derived class
void Classify::test(const Mat & testSet, LABEL_TYPE & predLabels)
{
	std::cout << "No classifier included for testing..." << std::endl;
}

int Classify::findBestParam(const Mat & dataSet, const Mat & labels, Mat & trainSetIndex,
							 const Mat & testSet, const LABEL_TYPE & trueLabels, LABEL_TYPE & predLabels)
{
	int nFreePara= modelParam->nFreeParam;

	const int npara=6;//<------ fix this
	
	if (nFreePara==0)
	{
		std::cout<<"No free parameters. Nothing to optimize."<<std::endl;
		goto exitLine;//return 0;
	}
	//if (nFreePara==1)
	//{
		// float * usePara_ptr= &modelParam->freePara_1.use;
		// float * allPara_ptr= &modelParam->freePara_1.all[0];
	//}
	//if (nFreePara==2)
	//{
		float * usePara1_ptr= &modelParam->freePara_1.use;
		float * allPara1_ptr= &modelParam->freePara_1.all[0];

		float * usePara2_ptr= &modelParam->freePara_2.use;
		float * allPara2_ptr= &modelParam->freePara_2.all[0];
	//}
	if (nFreePara>2)
	{
		std::cerr<<"# of free parameter is not implemented!"<<std::endl;
		goto exitLine;//return 0;
	}
	//
	// run the cross-validation: loop over the parameter grid
	//

	int maxIdx[2]={-1,-1}; // index for max. accu/best performance.

	switch (nFreePara)
	{
		case 1:
		{	
			float accuAll[npara]={0};
			
			//for(int ipara=0; ipara<npara; ++ipara)
			//{		
			//	*usePara_ptr = allPara_ptr[ipara];
			//	train( dataSet, labels, trainSetIndex);
			//	test( testSet, predLabels);
			//	
			//	//calculate accuracy
			//	accuAll[ipara] = calcClassPerf(trueLabels, predLabels); 
			//	std::cout << "Para="<< *usePara_ptr <<": Accuracy= " << accuAll[ipara] << std::endl;
			//}		
			//
			//// find parameter index with best performance
			//maxPerf(maxIdx, accuAll, npara, 1); 

			//*usePara_ptr = allPara_ptr[maxIdx[0]]; // use best parameter, update structure 
			//std::cout << ">>>> Best para="<< *usePara_ptr <<": Accuracy= " << accuAll[maxIdx[0]] << std::endl;
			
			std::cout << "Using Cross-Validation " << std::endl;
			
			for(int ipara=0; ipara<npara; ++ipara)
			{
				*usePara1_ptr = allPara1_ptr[ipara];
				accuAll[ipara]= CrossValid( dataSet, labels, trainSetIndex);
			}
			maxPerf(maxIdx, accuAll, npara, 1);

			*usePara1_ptr = allPara1_ptr[maxIdx[0]];
		}
		break;

		case 2:
		{
			float accuAll2[npara*npara];

			//for(int ipara1=0; ipara1<npara; ++ipara1)
			//{
			//	*usePara1_ptr = allPara1_ptr[ipara1];
			//	for(int ipara2=0; ipara2<npara; ++ipara2)
			//	{
			//		*usePara2_ptr = allPara2_ptr[ipara2];
			//		//train and test with this set of parameters
			//		train( dataSet, labels, trainSetIndex);
			//		test(  testSet, predLabels);
			//		//calculate accuracy
			//		accuAll2[ipara1*npara+ipara2] = calcClassPerf(trueLabels, predLabels); 
			//		std::cout << "Para1=" << std::setw(6) << *usePara1_ptr 
			//				  << " Para2="<< std::setw(7) << *usePara2_ptr 
			//				  << ": Accuracy= " << std::setw(7)<< accuAll2[ipara1*npara+ipara2] << std::endl;
			//	}
			//}

			//// find parameter set with best performance
			//
			//maxPerf(maxIdx, accuAll2, npara, npara); 
			//
			//// use best parameter, update structure 
			//*usePara1_ptr = allPara1_ptr[ maxIdx[0] ]; 
			//*usePara2_ptr = allPara2_ptr[ maxIdx[1] ];
			//	
			//std::cout << ">>>> Best para= "<< *usePara1_ptr <<" "<<*usePara2_ptr<<": Accuracy= " << accuAll2[maxIdx[0]*npara+maxIdx[1]] << std::endl;

			std::cout << "using cross-validation " << std::endl;
			
			for(int ipara1=0; ipara1<npara; ++ipara1)
			{
				*usePara1_ptr = allPara1_ptr[ipara1];
				for(int ipara2=0; ipara2<npara; ++ipara2)
				{
					*usePara2_ptr = allPara2_ptr[ipara2];
					accuAll2[ipara1*npara+ipara2]= CrossValid( dataSet, labels, trainSetIndex);
				}
			}
			
			maxPerf(maxIdx, accuAll2, npara, npara); 

			// use best parameters
			*usePara1_ptr = allPara1_ptr[ maxIdx[0] ]; 
			*usePara2_ptr = allPara2_ptr[ maxIdx[1] ];
		} 
		break;

		default:
			std::cout << "Number of free parameters is not implemented." << std::endl;
			break;
	}
exitLine:
	return 0;
}

float Classify::CrossValid( const Mat & dataSet,const Mat & labels, Mat & trainSetIndex)
{
	// performs cross-validation to estimate a classifier's performance for a set of parameters 
	const int nsub = 10; // number of subsamples, i.e. splits
	int ntrain = trainSetIndex.rows; // size of training set that is available for the cross-validation
	
	float cvTrainRatio = 0.80f; // fraction of data used for CV-training, rest for validation
	int n_cvTrain = (int)(cvTrainRatio*ntrain);
	int n_cvValid  = ntrain - n_cvTrain;

	setIdxT idx;
	idx.train= Mat::zeros(n_cvTrain,1,CV_32SC1) - 1;
	idx.test = Mat::zeros(n_cvValid,1,CV_32SC1) - 1;

	Mat cvTrainSetIdx = Mat::zeros(n_cvTrain,1,CV_32SC1) - 1;

	float accuAll[nsub];
	int idat,isub;

	for(isub=0; isub<nsub; ++isub)
	{
		// create index to split the training set in a CV-training and validation set
		splitDataSet(&idx, ntrain, cvTrainRatio);

		// set up the cross-validation training set index
		for(idat=0; idat<n_cvTrain; ++idat)
		{
			cvTrainSetIdx.at<int>(idat) = trainSetIndex.at<int>( idx.train.at<int>(idat) );
		}
	
		// set up validation/test set with labels
		Mat validSet = Mat::zeros( n_cvValid, dataSet.cols, CV_32FC1);
		LABEL_TYPE  validLabel_true(n_cvValid,0);
		LABEL_TYPE  validLabel_pred(n_cvValid,0);

		int tempIdx;
		for(idat=0; idat<n_cvValid; ++idat)
		{
			tempIdx = trainSetIndex.at<int>( idx.test.at<int>(idat) );
			validSet.row(idat) = dataSet.row( tempIdx ) + 0; //has to be expression to work with .row!
			validLabel_true[idat] = labels.at<float>( tempIdx );
		}

		// train the classifier with CV-training set for a set of parameters
		// test classifier using the validation set and calculate the performance
		
		train( dataSet, labels, cvTrainSetIdx);
		test( validSet, validLabel_pred);
					
		//calculate accuracy
		accuAll[isub] = calcClassPerf(validLabel_true, validLabel_pred);
		//std::cout << accuAll[isub] <<" ";
	}

	// calculate the average performance
	float meanAccu=0, stdAccu=0;
	for(isub=0; isub<nsub; ++isub)
		meanAccu+= accuAll[isub];
	
	meanAccu/=(float)nsub;

	for(isub=0; isub<nsub; ++isub)
		stdAccu+= (accuAll[isub]-meanAccu)*(accuAll[isub]-meanAccu);

	stdAccu  = sqrt( stdAccu/(float)nsub );
	
	std::cout <<"m= " <<  meanAccu <<" s= " <<  stdAccu << std::endl;
	return ( meanAccu-stdAccu );
}

int Classify::maxPerf(int * maxIdx, float * accu, int r, int c)
{
	int ir, ic;
	float maxVal = -1;

	for(ir=0;ir<r;++ir)
	{
		for(ic=0;ic<c;++ic)
			std::cout << accu[ir*c+ic] << " ";
		std::cout << std::endl;
	}

	for(ir=0;ir<r;++ir)
	{
		for(ic=0;ic<c;++ic)
			if (accu[ir*c+ic]>maxVal)
			{
				maxVal = accu[ir*c+ic];
				maxIdx[0]=ir;
				maxIdx[1]=ic;
			}
	}
	std::cout << "-> " << maxVal << " "<< maxIdx[0] << " " << maxIdx[1] <<std::endl;
	return 0;
}

void Classify::splitDataSet(setIdxT * idx,const int ndata, const float trainRatio)
{
	int tempIdx=0, idat;
	int ntrain = (int)(trainRatio*ndata);
	int ntest  =  ndata - ntrain;

	bool * isPicked = new bool[ndata];
	for(idat=0;idat<ndata;++idat)
		isPicked[idat]=false;

	for(idat=0;idat<ntrain;++idat)
	{
		do 
		{
			tempIdx = int(ceil(ndata*double(rand())/RAND_MAX*0.999999));
		} while(isPicked[tempIdx]);
		isPicked[tempIdx]= true;
		idx->train.at<int>(idat)=tempIdx;
	}
	
	int cnt=0;
	for(idat=0;idat<ndata;++idat)
	{
		if(!isPicked[idat])
			idx->test.at<int>(cnt++)=idat;
	}
	delete isPicked;
}


float Classify::calcClassPerf(const std::vector<float> & trueLabel,const std::vector<float> & predLabel)
{
	// calculate the classification performance
	float TP=0, TN=0, FP=0, FN=0, accu;

	for(long idat=0; idat<(long)trueLabel.size();++idat)
	{
		if     ( trueLabel[idat]==0 && predLabel[idat]==0)
			TN++;
		else if( trueLabel[idat]==1 && predLabel[idat]==1)
			TP++;
		else if( trueLabel[idat]==1 && predLabel[idat]==0)
			FN++;
		else if( trueLabel[idat]==0 && predLabel[idat]==1)
			FP++;
	}
	accu=(TN+TP)/(float)trueLabel.size();
	//std::cout << TN << " " << TP << " " << FN << " " << FP<< std::endl;

	return accu;
}

//
// auxiliary functions
//

void createRanDataSet(ClassData & cdat)
{
	// assign random feature values drawn from random distribution
	// create two gaussian clusters 
	std::cout<<" label     : "<< cdat.getLabel(3)<< std::endl;	
	std::cout<<" data      : "<< cdat.getData(0,2)<< std::endl;
	
	randn(cdat.dset,0,10);

	// create two Gauss Clusteres: shift one half and assign label
	int idat,idim;

	for(idat=cdat.nele/2; idat<cdat.nele;++idat)
	{
		cdat.setLabel(idat, 1);
		for(idim=0;idim<cdat.ndim;++idim)
		{
			//shift cluster
			cdat.setData(idat,idim, 15+ cdat.getData(idat,idim) );
		}
	}

	//check
	std::cout<<" label     : "<< cdat.getLabel(3)<< std::endl;	
	std::cout<<" data      : "<< cdat.getData(0,2)<< std::endl;
}