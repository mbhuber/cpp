#include <iostream>
#include "embed2.h"
//#include "/usr/include/armadillo"
// 

//using namespace arma;


int main(){
	Embed emb(10);
	emb.showSize();
	emb.showEmbData();
	
	Embed e;
	e.showSize();
	e.showEmbData();
	
	XOM x1;
	x1.showSize();
	x1.showEmbData();
	x1.showParam();
	
	XOM x2(1000,20,0.02);
	x2.showSize();
	x2.showEmbData();
	x2.showParam();
	x2.initRand();
	x2.showEmbData();
	
	std::cout << x2.findWinnerNeuron() << std::endl;
	std::cout << x2.findWinnerNeuron() << std::endl;
	std::cout << x2.findWinnerNeuron() << std::endl;
	
	/*
	 // directly specify the matrix size (elements are uninitialised)
	 mat A(2,3);
	 
	 // .n_rows = number of rows    (read only)
	 // .n_cols = number of columns (read only)
	 cout << "A.n_rows = " << A.n_rows << endl;
	 cout << "A.n_cols = " << A.n_cols << endl;
	 */
	
	return 0;
}