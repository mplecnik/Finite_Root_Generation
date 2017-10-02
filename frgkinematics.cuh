#ifndef FRGKINEMATICS_CUH
#define FRGKINEMATICS_CUH

#include "htpykinematics.cuh"
#include <curand_kernel.h>

__device__ void GenerateStart(curandState, Vect<nV>*, Vect<nP>*);

__device__ void CognateSolns(Vect<nV>*, Vect<nP>*, Vect<nV>*);

#endif