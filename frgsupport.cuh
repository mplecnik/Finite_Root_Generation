#ifndef FRGSUPPORT_CUH
#define FRGSUPPORT_CUH

#include <deque>
#include "htpykinematics.cuh"

void SortingDeletingMerging(std::deque<Vect<nV>>*, Vect<nV>*, int*, int*, int);

Real LambertW(Real);

#endif