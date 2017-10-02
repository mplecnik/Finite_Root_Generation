#ifndef HTPYSUPPORT_CUH
#define HTPYSUPPORT_CUH

#include "htpykinematics.cuh"

__device__ void SwitchPatch(Vect<nV + 2>*, Vect<nV + 2>*, Vect<nV + 2>*);

__device__ void q(Real, Vect<nP>*, Vect<nP>*, Cplx*, Vect<nP>*);

__device__ void dqdt(Real, Vect<nP>*, Vect<nP>*, Cplx*, Vect<nP>*);

__device__ void H(Vect<nV + 2>*, Vect<nV + 2>*, Vect<nP>*, Vect<nP>*, Cplx*, Vect<nV + 1>*);

__device__ void JH(Vect<nV + 2>*, Vect<nV + 2>*, Vect<nP>*, Vect<nP>*, Cplx*, Matr<nV + 1, nV + 2>*);

__device__ void IVP(bool, Vect<nV + 2>*, Vect<nV + 2>*, Vect<nV + 2>*,
	Vect<nP>*, Vect<nP>*, Cplx*, Vect<nV + 2>*);

__device__ void Newton(Real, Vect<nV + 2>*, Vect<nV + 2>*,
	Vect<nP>*, Vect<nP>*, Cplx*, int, Real, int, Vect<nV + 2>*, Real*);


#endif