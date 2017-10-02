#ifndef HTPYKINEMATICS_CUH
#define HTPYKINEMATICS_CUH

#include "classes.cuh"
#include <cublas_v2.h>


#define nV 21 //number of variables
#define nP 30 //number of parameters

__device__ void FSYS(Vect<nV + 1>*, Vect<nP>*, Vect<nV>*);

__device__ void JVAR(Vect<nV + 1>*, Vect<nP>*, Matr<nV, nV + 1>*);

__device__ void JPAR(Vect<nV + 1>*, Vect<nP>*, Matr<nV, nP>*);

#endif