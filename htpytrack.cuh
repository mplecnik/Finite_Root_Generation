#ifndef HTPYTRACK_CUH
#define HTPYTRACK_CUH

#include "htpykinematics.cuh"
#include <curand_kernel.h>

__device__ void TrackPath(Vect<nV>*, Vect<nP>*, Vect<nP>*, Cplx*, curandState, Vect<nV + 2>*, int*);

#endif