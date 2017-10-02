#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "htpykinematics.cuh"

// Constants
#define UpperRand 5. // upper limit of randomly generated numbers
#define LowerRand -5. // lower limit of randomly generated numbers




__device__ void GenerateStart(curandState randState, Vect<nV> *z, Vect<nP> *q){

	// Identifying the variable z
	Cplx *A = &z[0].vals[0];
	Cplx *B = &z[0].vals[1];
	Cplx *C = &z[0].vals[2];
	Cplx *D = &z[0].vals[3];
	Cplx *F = &z[0].vals[4];
	Cplx *G = &z[0].vals[5];
	Cplx *H = &z[0].vals[6];
	Cplx *Ac = &z[0].vals[7];
	Cplx *Bc = &z[0].vals[8];
	Cplx *Cc = &z[0].vals[9];
	Cplx *Dc = &z[0].vals[10];
	Cplx *Fc = &z[0].vals[11];
	Cplx *Gc = &z[0].vals[12];
	Cplx *Hc = &z[0].vals[13];
	Cplx *U = &z[0].vals[14-1]; // subtract 1 for proper indexing from 1,...,7

	// Identifying the parameter q
	Cplx *P = &q[0].vals[0];
	Cplx *Pc = &q[0].vals[8];
	Cplx *Q = &q[0].vals[16-1]; // subtract 1 for proper indexing from 1,...,7
	Cplx *Qc = &q[0].vals[23-1]; // subtract 1 for proper indexing from 1,...,7

	// Generate random numbers for A, B, C, D, F, G, H, Ac, Bc, Cc, Dc, Fc, Gc, Hc 
#pragma unroll
	for (int i = 0; i < 14; ++i){
		z[0].vals[i].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
		z[0].vals[i].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	}

	// Generate random numbers for P0, Pc0
	P[0].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	P[0].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	Pc[0].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	Pc[0].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;

	Cplx *RUc = P; // RUc shares space with P

	// Generate random numbers for RUc
#pragma unroll
	for (int j = 1; j <= 7; ++j){
		RUc[j].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
		RUc[j].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	}

#pragma unroll
	for (int j = 1; j <= 7; ++j){

		Cplx A_quad = cplxMul(cplxSub(*G, *D), cplxSub(cplxMul(cplxDiv({ 1, 0 }, RUc[j]), cplxSub(*Hc, *Gc)), cplxSub(*Hc, *Fc)));
		Cplx Ac_quad = cplxMul(cplxSub(*Gc, *Dc), cplxSub(cplxMul(RUc[j], cplxSub(*H, *G)), cplxSub(*H, *F)));
		Cplx B_quad = cplxAdd(
			cplxSub(cplxSub({ 0, 0 }, cplxMul(RUc[j], cplxMul(cplxSub(*H, *G), cplxSub(*Hc, *Fc)))),
			cplxMul(cplxDiv({ 1, 0 }, RUc[j]), cplxMul(cplxSub(*Hc, *Gc), cplxSub(*H, *F)))),
			cplxAdd(cplxMul(cplxSub(*G, *D), cplxSub(*Gc, *Dc)),
			cplxAdd(cplxMul(cplxSub(*H, *G), cplxSub(*Hc, *Gc)),
			cplxSub(cplxMul(cplxSub(*H, *F), cplxSub(*Hc, *Fc)),
			cplxMul(cplxSub(*F, *D), cplxSub(*Fc, *Dc)))))
			);

		Real pm = 1.; if (curand_uniform_double(&randState) < 0.5){ pm = -1.; }

		Cplx TUc = cplxMul(
			cplxDiv({ 1, 0 }, cplxMul({ 2, 0 }, A_quad)),
			cplxAdd(cplxSub({ 0, 0 }, B_quad), 
			cplxMul({ pm, 0 }, cplxSqrt(cplxSub(cplxMul(B_quad, B_quad), cplxMul({ 4, 0 }, cplxMul(A_quad, Ac_quad))))))
			);

		Cplx SUc = cplxMul(
			cplxDiv({ 1, 0 }, cplxSub(*F, *D)), 
			cplxAdd(cplxMul(TUc, cplxSub(*G, *D)), cplxSub(cplxMul(RUc[j], cplxSub(*H, *G)), cplxSub(*H, *F)))
			);

		A_quad = cplxMul(
			cplxSub(*Ac, *Bc), 
			cplxSub(cplxSub(cplxMul(RUc[j], cplxSub(*H, *C)), cplxMul(SUc, cplxSub(*F, *B))), cplxSub(*H, *F))
			);
		Ac_quad = cplxMul(
			cplxSub(*A, *B),
			cplxSub(cplxSub(cplxMul(cplxDiv({ 1, 0 }, RUc[j]), cplxSub(*Hc, *Cc)), cplxMul(cplxDiv({ 1, 0 }, SUc), cplxSub(*Fc, *Bc))), cplxSub(*Hc, *Fc))
			);
		B_quad = cplxAdd(cplxMul(
			cplxSub(cplxSub(cplxMul(RUc[j], cplxSub(*H, *C)), cplxMul(SUc, cplxSub(*F, *B))), cplxSub(*H, *F)),
			cplxSub(cplxSub(cplxMul(cplxDiv({ 1, 0 }, RUc[j]), cplxSub(*Hc, *Cc)), cplxMul(cplxDiv({ 1, 0 }, SUc), cplxSub(*Fc, *Bc))), cplxSub(*Hc, *Fc))),
			cplxSub(cplxMul(cplxSub(*A, *B), cplxSub(*Ac, *Bc)), cplxMul(cplxSub(*C, *A), cplxSub(*Cc, *Ac)))
			);

		pm = 1.; if (curand_uniform_double(&randState) < 0.5){ pm = -1.; }

		U[j] = cplxMul(
			cplxDiv({ 1, 0 }, cplxMul({ 2, 0 }, A_quad)),
			cplxAdd(cplxSub({ 0, 0 }, B_quad),
			cplxMul({ pm, 0 }, cplxSqrt(cplxSub(cplxMul(B_quad, B_quad), cplxMul({ 4, 0 }, cplxMul(A_quad, Ac_quad))))))
			);

		Cplx QUc = cplxMul(
			cplxDiv({ -1, 0 }, cplxSub(*C, *A)), 
			cplxAdd(cplxMul(cplxDiv({ 1, 0 }, U[j]), cplxSub(*A, *B)), cplxSub(cplxSub(cplxMul(RUc[j], cplxSub(*H, *C)), cplxMul(SUc, cplxSub(*F, *B))), cplxSub(*H, *F)))
			);

		Q[j] = cplxMul(QUc, U[j]);
		Qc[j] = cplxDiv({ 1, 0 }, Q[j]);

		Cplx S = cplxMul(SUc, U[j]);

		P[j] = cplxAdd(*B, cplxAdd(cplxMul(S, cplxSub(*F, *B)), cplxMul(U[j], cplxSub(P[0], *F))));
		Pc[j] = cplxAdd(*Bc, cplxAdd(cplxMul(cplxDiv({ 1, 0 }, S), cplxSub(*Fc, *Bc)), cplxMul(cplxDiv({ 1, 0 }, U[j]), cplxSub(Pc[0], *Fc))));

	}


	//memset(z[0].vals, 0, sizeof(Vect<nV>));

	//for (int i = 0; i < 20; ++i){

	//	Real pm = 1.; if (curand_uniform_double(&randState) < 0.5){ pm = -1.; }

	//	z[0].vals[i] = {pm, 0};

	//}


}



__device__ void Cognate34(Cplx *orig, Cplx P, Cplx *cogn){

	// The 3/4 cognate for a Stephenson II path generator (from Dijksman)
	// orig: cognate generator A, B, C, D, F, G, H
	// P aka P0
	// cogn: output cognate A, B, C, D, F, G, H

	Cplx A = orig[0];
	Cplx B = orig[1];
	Cplx C = orig[2];
	Cplx D = orig[3];
	Cplx F = orig[4];
	Cplx G = orig[5];
	Cplx H = orig[6];


	Cplx *Ap = &cogn[0];
	Cplx *Bp = &cogn[1];
	Cplx *Cp = &cogn[2];
	Cplx *Gp = &cogn[5];
	Cplx scalar = cplxDiv(cplxSub(H, G), cplxSub(C, G));
	*Ap = cplxAdd(cplxMul(scalar, cplxSub(A, D)), D);
	*Bp = cplxAdd(cplxMul(scalar, cplxSub(B, D)), D);
	*Cp = cplxAdd(cplxMul(scalar, cplxSub(C, D)), D);
	*Gp = cplxAdd(D, cplxSub(H, G));


	Cplx *App = &cogn[0];
	Cplx *Cpp = &cogn[2];
	Cplx *Dp = &cogn[3];
	Cplx *Gpp = &cogn[5];
	Cplx *Hp = &cogn[6];
	scalar = cplxDiv(cplxSub(B, F), cplxSub(*Bp, F));
	*App = cplxAdd(cplxMul(scalar, cplxSub(*Ap, F)), F);
	*Cpp = cplxAdd(cplxMul(scalar, cplxSub(*Cp, F)), F);
	*Dp = cplxAdd(cplxMul(scalar, cplxSub(D, F)), F);
	*Gpp = cplxAdd(cplxMul(scalar, cplxSub(*Gp, F)), F);
	*Hp = cplxAdd(cplxMul(scalar, cplxSub(H, F)), F);

	cogn[1] = B;
	cogn[4] = F;

}


__device__ void Cognate12(Cplx *orig, Cplx P, Cplx *cogn){

	// The 1/2 cognate for a Stephenson II path generator (from Dijksman)
	// orig: cognate generator A, B, C, D, F, G, H
	// P aka P0
	// cogn: output cognate A, B, C, D, F, G, H

	Cplx A = orig[0];
	Cplx B = orig[1];
	Cplx C = orig[2];
	Cplx D = orig[3];
	Cplx F = orig[4];
	Cplx G = orig[5];
	Cplx H = orig[6];


	Cplx *Dp = &cogn[3];
	Cplx *Gp = &cogn[5];
	Cplx scalar = cplxDiv(cplxSub(B, F), cplxSub(D, F));
	*Dp = cplxAdd(cplxMul(scalar, cplxSub(D, H)), H);
	*Gp = cplxAdd(cplxMul(scalar, cplxSub(G, H)), H);


	Cplx *Ap = &cogn[0];
	Cplx *Cp = &cogn[2];
	Cplx *Dpp = &cogn[3];
	Cplx *Fpp = &cogn[4];
	Cplx *Gpp = &cogn[5];
	Cplx *Hp = &cogn[6];
	scalar = cplxDiv(cplxSub(P, F), cplxSub(H, F));
	*Ap = cplxAdd(cplxMul(scalar, cplxSub(A, B)), B);
	*Cp = cplxAdd(cplxMul(scalar, cplxSub(C, B)), B);
	*Dpp = cplxAdd(cplxMul(scalar, cplxSub(*Dp, B)), B);
	*Fpp = cplxAdd(B, cplxSub(P, F));
	*Gpp = cplxAdd(cplxMul(scalar, cplxSub(*Gp, B)), B);
	*Hp = cplxAdd(cplxMul(scalar, cplxSub(H, B)), B);

	cogn[1] = B;

}


__device__ void CognateSolns(Vect<nV> *z, Vect<nP> *q, Vect<nV> *cognSolns){

	// z: an nonhomogeneous solution
	// q: parameters
	// cognSolns: a list of all cognate solutions

	// Identify B and F
	Cplx B = z[0].vals[1];
	Cplx F = z[0].vals[4];

	// Identify P's
	Cplx *P = &q[0].vals[0];
	Cplx *Pc = &q[0].vals[8];

	// Identify output cognates
	cognSolns[0] = z[0];
	Vect<nV> *cogn12soln = &cognSolns[1];
	//Vect<nV> *cogn34soln = &cognSolns[2];
	//Vect<nV> *cogn1234soln = &cognSolns[3];

	// Identify Uup, compute Udown
	Cplx *Uup = &z[0].vals[14 - 1]; // minus 1 for proper indices
	Cplx *Udown = &cogn12soln[0].vals[14-1]; // minus 1 for proper indices

#pragma unroll
	for (int i = 1; i <= 7; ++i){
		Udown[i] = cplxDiv(cplxSub(cplxSub(P[i], B), cplxMul(Uup[i], cplxSub(P[0], F))), cplxSub(F, B));
		//cogn34soln[0].vals[13 + i] = Uup[i];
		//cogn1234soln[0].vals[13 + i] = Udown[i];
	}

	// Perform cognate calcs
	Cognate12(&z[0].vals[0], P[0], &cogn12soln[0].vals[0]);
	Cognate12(&z[0].vals[7], Pc[0], &cogn12soln[0].vals[7]);

	//Cognate34(&z[0].vals[0], P[0], &cogn34soln[0].vals[0]);
	//Cognate34(&z[0].vals[7], Pc[0], &cogn34soln[0].vals[7]);

	//Cognate34(&cogn12soln[0].vals[0], P[0], &cogn1234soln[0].vals[0]);
	//Cognate34(&cogn12soln[0].vals[7], Pc[0], &cogn1234soln[0].vals[7]);

	// find cognate with the smallest real component of the first element
	int minpos = 0;
#pragma unroll
	for (int i = 1; i < 2; ++i){
		if (cognSolns[i].vals[0].x < cognSolns[minpos].vals[0].x){ minpos = i; }
	}

	// switch cognate ordering so the smaller one appears first
	if (minpos == 1){ 
#pragma unroll
		for (int i = 0; i < nV; ++i){
			Cplx temp = cognSolns[0].vals[i];
			cognSolns[0].vals[i] = cognSolns[1].vals[i];
			cognSolns[1].vals[i] = temp;
		}
	}

}

