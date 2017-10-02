#ifndef RUNGEKUTTA_CUH
#define RUNGEKUTTA_CUH

#include "htpykinematics.cuh"

__device__ void DormandPrince65(Real, Real, bool, Vect<nV + 2>*, Vect<nV + 2>*,
	Vect<nV + 2>*, Vect<nP>*, Vect<nP>*, Cplx*, Vect<nV + 2>*, Vect<nV + 2>*, Real*, Real*);

#endif