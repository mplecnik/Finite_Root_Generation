#include <cublas_v2.h>
#include "classes.cuh"


__device__ Cplx cplxPow(Cplx x, int n){
	Cplx y = x;
#pragma unroll 1
	for (int i = 1; i < n; ++i){
		y = cplxMul(y, x);
	}
	//int i = 1;
	//Cplx y = x;
	//while (i < n){ y = cplxMul(y, x); i++; }
	return y;
}

__device__ Cplx cplxPow(Cplx x, Cplx n){
	Cplx y = x;
#pragma unroll 1
	for (int i = 1; i < n.x; ++i){
		y = cplxMul(y, x);
	}
	return y;
}

