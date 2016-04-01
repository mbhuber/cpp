/*
 *  embed2.h
 *  embed_02
 *
 *  Created by Markus on 8/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef EMBED_H_
#define EMBED_H_

namespace embType {
	typedef unsigned long lType;
}

//using embType::lType;
using namespace embType;

class Embed
{
protected:
	lType nele; // elements of data
    unsigned dDim; // dimensions of data
	float * dat;        // data
	float * emb;        // embedding
	unsigned eDim;// embedding dimensionality
    
public:
	Embed();
	Embed(lType n);
	~Embed();
	void showSize();
	void showEmbData();
};

class XOM : public Embed {
private:	
	float sig1;
	float sig2;

public:
	XOM();
	XOM(lType n, float s1, float s2);
	//~XOM();
	void showParam();
	void initRand();
	lType findWinnerNeuron();
};

#endif
