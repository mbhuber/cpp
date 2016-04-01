/*
 *  embed_02.cpp
 *  embed_02
 *
 *  Created by Markus on 8/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "embed2.h"
#include <iostream>
#include <cstdlib>

using embType::lType;

// Embed methods

Embed::Embed(){
	dDim=3;
	nele=10;
	eDim=2;
	dat = new float [nele*dDim];
	emb = new float [nele*eDim];
	
	// create random data set
	for (lType i=0;i<nele*dDim;i++) dat[i]= float(rand())/RAND_MAX*10;
	//for (lType i=0;i<nele*eDim;i++) emb[i]= 0;
}

Embed::Embed(lType n){
	dDim=3;
	nele=n;
	eDim=2;
	dat = new float [nele*dDim];
	emb = new float [nele*eDim];
	
	// create random data set
	for (lType i=0;i<nele*dDim;i++) dat[i]= float(rand())/RAND_MAX*10;
	//for (lType i=0;i<nele*eDim;i++) emb[i]= 0;
}

Embed::~Embed(){
	delete dat;
	delete emb;
}

void Embed::showSize(){
	std::cout << "Data: dim= " << dDim <<" "<< ", ele: " << nele << std::endl;
	std::cout << "Emb.: dim= " << eDim <<" "<< ", ele: " << nele << std::endl;	
}

void Embed::showEmbData(){
	std::cout << "Data: 1.   ele.: " << dat[0]
	          << ", last ele.: " << dat[nele*dDim-1] << std::endl;
	std::cout << "Emb.: 1.   ele.: " << emb[0]
			  << ", last ele.: " << emb[nele*eDim-1] << std::endl;
}

// XOM methods

XOM::XOM() : Embed(1) {
	sig1=10;
	sig2=0.05;
}

XOM::XOM(unsigned long n, float s1, float s2) : Embed(n) {
	sig1=s1;
	sig2=s2;
}

//XOM::~XOM();

void XOM::showParam(){
	std::cout << "sigma 1: " << sig1 << std::endl;
	std::cout << "sigma 2: " << sig2 << std::endl;
}

// initialize embedding with random values
void XOM::initRand(){
	for (lType i=0;i<nele*eDim;i++) emb[i]= float(rand())/RAND_MAX;
}

// select random point in feature space and find winner neuron
lType XOM::findWinnerNeuron(){
	//
    // MAP Space
    //
	
	// select a random point in the exploration space
	lType idim;
	float mX[eDim];
	for(idim=0;idim<eDim;idim++){
		mX[idim] = float(rand())/RAND_MAX;// random values for each dimension
	}
    
    // distance between random input vector and all weight vectors on the
    // map; find minimum/winner neuron
    lType iele, minIdx;
    float mapDist,minDist=1e20,dumDist;
	
    for(iele=0;iele<nele;iele++){
        // calculate distance between random point and one map point 
        mapDist=0.;
        for(idim=0;idim<eDim;idim++){
            dumDist= mX[idim]- emb[iele*eDim+idim];
            mapDist= mapDist + dumDist*dumDist;
        }
        // check if temp. distance is minimum so far
        if(mapDist<minDist){
            minDist= mapDist; // save new min distance and
            minIdx= iele;  // and index of winner neuron
        }
    }
	return minIdx;
}