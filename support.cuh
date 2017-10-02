#ifndef SUPPORT_CUH
#define SUPPORT_CUH

#include <cublas_v2.h>
#include "classes.cuh"

__device__ Cplx cplxPow(Cplx, int);

__device__ Cplx cplxPow(Cplx, Cplx);

#endif