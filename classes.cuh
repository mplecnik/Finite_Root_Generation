#ifndef CLASSES_CUH
#define CLASSES_CUH

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>


//#define SINGLE_PREC // uncomment this line for single precision

#ifdef SINGLE_PREC
typedef float Real;
typedef cuComplex Cplx;
__host__ __device__ static __inline__ Cplx cplxAdd(Cplx x, Cplx y){ return cuCaddf(x, y); }
__host__ __device__ static __inline__ Cplx cplxSub(Cplx x, Cplx y){ return cuCsubf(x, y); }
__host__ __device__ static __inline__ Cplx cplxMul(Cplx x, Cplx y){ return cuCmulf(x, y); }
__host__ __device__ static __inline__ Cplx cplxDiv(Cplx x, Cplx y){ return cuCdivf(x, y); }
__host__ __device__ static __inline__ Cplx cplxFma(Cplx x, Cplx y, Cplx d){ return cuCfmaf(x, y, d); }
__host__ __device__ static __inline__ Real cplxAbs(Cplx x){ return cuCabsf(x); }
//const auto& __host__ __device__ __inline__  mul = cuCmulf;
//__host__ __device__ static __inline__ Cplx (&mul)(Cplx, Cplx) = cuCmulf;
//#define mul cuCmulf
//typedef Cplx(*mul_alias)(Cplx, Cplx);
//mul_alias mul = cuCmulf;
#else
typedef double Real;
typedef cuDoubleComplex Cplx;
__host__ __device__ static __inline__ Cplx cplxAdd(Cplx x, Cplx y){ return cuCadd(x, y); }
__host__ __device__ static __inline__ Cplx cplxSub(Cplx x, Cplx y){ return cuCsub(x, y); }
__host__ __device__ static __inline__ Cplx cplxMul(Cplx x, Cplx y){ return cuCmul(x, y); }
__host__ __device__ static __inline__ Cplx cplxDiv(Cplx x, Cplx y){ return cuCdiv(x, y); }
__host__ __device__ static __inline__ Cplx cplxFma(Cplx x, Cplx y, Cplx d){ return cuCfma(x, y, d); }
__host__ __device__ static __inline__ Real cplxAbs(Cplx x){ return cuCabs(x); }
//const auto& __device__ mul = cuCmul;
//Cplx(*mul)(Cplx, Cplx) = cuCmul;
//#define mul cuCmul
#endif

__host__ __device__ static __inline__ Cplx cplxSqrt(Cplx x){
	double r = cplxAbs(x);
	double cosA = x.x / r;
	Cplx out = { sqrt(r * (cosA + 1.0) / 2.0), sqrt(r * (1.0 - cosA) / 2.0) };
	if (signbit(x.y)){ out.y *= -1.0; }	// signbit should be false if x.y is negative
	return out;
}



// class Vect and its member functions


template<int M>
class __align__(16) Vect{
public:
	Cplx vals[M];
	__host__ __device__ Vect();
	__host__ __device__ Vect(Cplx *v);
	__host__ __device__ void set(Cplx *v);
	__host__ __device__ Real norm();
	__host__ __device__ Vect<M> scale(Real s);
	__host__ __device__ Vect<M> scale(Cplx s);
	__host__ __device__ Vect<M> add(Vect<M> vect2);
	__host__ __device__ Vect<M> sub(Vect<M> vect2);
	__host__ __device__ Cplx dot(Vect<M> vect2);
	__host__ void print();
};

template<int M>
__host__ __device__ Vect<M>::Vect(){}

template<int M>
__host__ __device__ Vect<M>::Vect(Cplx *v){
	for (int i = 0; i < M; ++i){ vals[i] = v[i]; }
}

template<int M>
__host__ __device__ void Vect<M>::set(Cplx *v){
	for (int i = 0; i < M; ++i){ vals[i] = v[i]; }
}

template<int M>
__host__ __device__ Real Vect<M>::norm(){
	Real out = 0;
	for (int i = 0; i < M; ++i){ 
		out += vals[i].x * vals[i].x + vals[i].y * vals[i].y;
	}
	return sqrt(out);
}

template<int M>
__host__ __device__ Vect<M> Vect<M>::scale(Real s){
	Vect<M> scaledVect;
	for (int i = 0; i < M; ++i){ scaledVect.vals[i] = cplxMul({ s, 0 }, vals[i]); }
	return scaledVect;
}

template<int M>
__host__ __device__ Vect<M> Vect<M>::scale(Cplx s){
	Vect<M> scaledVect;
	for (int i = 0; i < M; ++i){ scaledVect.vals[i] = cplxMul(s, vals[i]); }
	return scaledVect;
}


template<int M>
__host__ __device__ Vect<M> Vect<M>::add(Vect<M> vect2){
	Vect<M> sum;
	for (int i = 0; i < M; ++i){ sum.vals[i] = cplxAdd(vals[i], vect2.vals[i]); }
	return sum;
}

template<int M>
__host__ __device__ Vect<M> Vect<M>::sub(Vect<M> vect2){
	Vect<M> diff;
	for (int i = 0; i < M; ++i){ diff.vals[i] = cplxSub(vals[i], vect2.vals[i]); }
	return diff;
}

template<int M>
__host__ __device__ Cplx Vect<M>::dot(Vect<M> vect2){
	Cplx prod = { 0, 0 };
	for (int i = 0; i < M; ++i){
		prod = cplxAdd(prod, cplxMul(vals[i], vect2.vals[i]));
	}
	return prod;
}

template<int M>
__host__ void Vect<M>::print(){
	for (int i = 0; i < M; ++i){
		std::cout << vals[i].x << " " << vals[i].y << "I" << std::endl;
	}
}


// class Matr and its member functions
template<int R, int C>
class __align__(16) Matr{
public:
	Cplx vals[R*C];
	__host__ __device__ Matr();
	__host__ __device__ Matr(Cplx *v);
	__host__ __device__ void set(Cplx *v);
	__host__ __device__ Cplx getElemB0(int i, int j);
	__host__ __device__ void setElemB0(int i, int j, Cplx z);
	__host__ __device__ Cplx getElemB1(int i, int j);
	__host__ __device__ void setElemB1(int i, int j, Cplx z);
	__host__ __device__ Vect<R> mul(Vect<C>);
	__host__ __device__ Matr<R,C> scale(Real s);
	__host__ __device__ Matr<R,C> scale(Cplx s);
	__host__ __device__ Matr<R,C> add(Matr<R,C> matr2);
	__host__ __device__ Matr<R,C> sub(Matr<R,C> matr2);
	__host__ void print();
};

template<int R, int C>
__host__ __device__ Matr<R,C>::Matr(){};

template<int R, int C>
__host__ __device__ Matr<R,C>::Matr(Cplx *v){
	for (int i = 0; i < R*C; ++i){ vals[i] = v[i]; }
}

template<int R, int C>
__host__ __device__ void Matr<R,C>::set(Cplx *v){
	for (int i = 0; i < R*C; ++i){ vals[i] = v[i]; }
}

template<int R, int C>
__host__ __device__ Cplx Matr<R,C>::getElemB0(int i, int j){
	return vals[C*i + j];
}

template<int R, int C>
__host__ __device__ void Matr<R,C>::setElemB0(int i, int j, Cplx z){
	vals[C*i + j] = z;
}

template<int R, int C>
__host__ __device__ Cplx Matr<R,C>::getElemB1(int i, int j){
	return vals[C*(i - 1) + (j - 1)];
}

template<int R, int C>
__host__ __device__ void Matr<R,C>::setElemB1(int i, int j, Cplx z){
	vals[C*(i - 1) + (j - 1)] = z;
}

template<int R, int C>
__host__ __device__ Vect<R> Matr<R, C>::mul(Vect<C> vect){
	Vect<R> prod;
	for (int i = 0; i < R; ++i){
		prod.vals[i] = { 0, 0 };
		for (int j = 0; j < C; ++j){
			prod.vals[i] = cplxAdd(prod.vals[i], cplxMul(vals[i*C + j], vect.vals[j]));
		}
	}
	return prod;
}

template<int R, int C>
__host__ __device__ Matr<R,C> Matr<R, C>::scale(Real s){
	Matr<R,C> scaledMatr;
	for (int i = 0; i < R*C; ++i){ scaledMatr.vals[i] = cplxMul({ s, 0 }, vals[i]); }
	return scaledMatr;
}

template<int R, int C>
__host__ __device__ Matr<R,C> Matr<R, C>::scale(Cplx s){
	Matr<R,C> scaledMatr;
	for (int i = 0; i < R*C; ++i){ scaledMatr.vals[i] = cplxMul(s, vals[i]); }
	return scaledMatr;
}

template<int R, int C>
__host__ __device__ Matr<R,C> Matr<R,C>::add(Matr<R,C> matr2){
	Matr<R,C> sum;
	for (int i = 0; i < R*C; ++i){ sum.vals[i] = cplxAdd(vals[i], matr2.vals[i]); }
	return sum;
}

template<int R, int C>
__host__ __device__ Matr<R,C> Matr<R,C>::sub(Matr<R,C> matr2){
	Matr<R,C> diff;
	for (int i = 0; i < R*C; ++i){ diff.vals[i] = cplxSub(vals[i], matr2.vals[i]); }
	return diff;
}

template<int R, int C>
__host__ void Matr<R,C>::print(){
	for (int i = 0; i < R; ++i){
		for (int j = 0; j < C; ++j){
			std::cout << vals[C*i + j].x << " " << vals[C*i + j].y << "I" << "   ";
		}
		std::cout << std::endl;
	}
}


#endif