#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "support.cuh"
#include <stdlib.h>
#include "htpykinematics.cuh"


__device__ void FSYS(Vect<nV + 1> *Z, Vect<nP> *q, Vect<nV> *eval){

	Cplx A = Z[0].vals[0];
	Cplx B = Z[0].vals[1];
	Cplx C = Z[0].vals[2];
	Cplx D = Z[0].vals[3];
	Cplx F = Z[0].vals[4];
	Cplx G = Z[0].vals[5];
	Cplx H = Z[0].vals[6];
	Cplx Ac = Z[0].vals[7];
	Cplx Bc = Z[0].vals[8];
	Cplx Cc = Z[0].vals[9];
	Cplx Dc = Z[0].vals[10];
	Cplx Fc = Z[0].vals[11];
	Cplx Gc = Z[0].vals[12];
	Cplx Hc = Z[0].vals[13];

	Cplx *U = &Z[0].vals[14-1]; // subtract 1 for indexing 1 to 7

	Cplx Z0 = Z[0].vals[21];

	Cplx *P = &q[0].vals[0]; // indexed from 0 to 7
	Cplx *Pc = &q[0].vals[8]; // indexed from 0 to 7

	Cplx *Q = &q[0].vals[16-1]; // subtract 1 for indexing 1 to 7
	Cplx *Qc = &q[0].vals[23-1]; // subtract 1 for indexing 1 to 7

#pragma unroll
	for (int j = 1; j <= 7; ++j){

		eval[0].vals[j - 1] = cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(\
			cplxMul({ -1, 0 }, cplxMul(cplxAdd(cplxMul(P[j], Pc[j]), cplxMul(\
			P[0], Pc[0])), cplxPow(Z0, 3))), cplxMul(cplxMul(cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j]), Z0), cplxAdd(cplxMul({ -1, 0 }, A \
			), cplxMul(P[j], Z0)))), cplxMul(cplxMul(\
			Ac, Z0), cplxAdd(cplxAdd(cplxMul({ -1, 0 }, A), C), cplxMul(P[j], Z0)) \
			)), cplxMul(cplxMul(Hc, Z0), cplxAdd(cplxMul({ -1, 0 }, C), cplxMul(\
			P[0], Z0)))), cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), \
			C), Q[j]), Z0), cplxAdd(cplxMul({ -1, 0 }, Ac), cplxMul(Pc[j], Z0)) \
			)), cplxMul(cplxMul(A, Z0), cplxAdd(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc \
			), cplxMul(Pc[j], Z0)))), cplxMul(cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(Pc[j], Z0)) \
			)), cplxMul(cplxMul(H, Z0), cplxAdd(cplxMul({ -1, 0 }, Cc), cplxMul(\
			Pc[0], Z0)))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, 2), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(Pc[0], Z0))), U[j]));

		eval[0].vals[j + 6] = cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxSub(cplxAdd(cplxMul(cplxMul(Bc, P[j])\
			, cplxPow(Z0, { 2, 0 })), cplxMul(cplxMul(B, Pc[j]), cplxPow(Z0, { 2, 0 }))), cplxMul(\
			cplxAdd(cplxMul(P[j], Pc[j]), cplxMul(P[0], Pc[0])), cplxPow(Z0, { 3, 0 } \
			))), cplxMul(cplxMul(Fc, Z0), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[0], Z0)))), cplxMul(cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(Pc[j], Z0)) \
			)), cplxMul(cplxMul(F, Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[0], Z0)))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(Pc[0], Z0))), U[j]));

		eval[0].vals[j + 13] = cplxSub(cplxAdd(cplxAdd(cplxSub(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, \
			0 }, D), G), cplxAdd(cplxMul({ -1, 0 }, Dc), Gc)), Z0 \
			), cplxMul(cplxMul(Z0, cplxAdd(cplxSub(cplxAdd(cplxSub(A, B \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(cplxMul(\
			U[j], cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j] \
			)), cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

	}

}


__device__ void JVAR(Vect<nV+1> *Z, Vect<nP> *q, Matr<nV,nV+1> *eval){

	Cplx A = Z[0].vals[0];
	Cplx B = Z[0].vals[1];
	Cplx C = Z[0].vals[2];
	Cplx D = Z[0].vals[3];
	Cplx F = Z[0].vals[4];
	Cplx G = Z[0].vals[5];
	Cplx H = Z[0].vals[6];
	Cplx Ac = Z[0].vals[7];
	Cplx Bc = Z[0].vals[8];
	Cplx Cc = Z[0].vals[9];
	Cplx Dc = Z[0].vals[10];
	Cplx Fc = Z[0].vals[11];
	Cplx Gc = Z[0].vals[12];
	Cplx Hc = Z[0].vals[13];

	Cplx *U = &Z[0].vals[14 - 1]; // subtract 1 for indexing 1 to 7

	Cplx Z0 = Z[0].vals[21];

	Cplx *P = &q[0].vals[0]; // indexed from 0 to 7
	Cplx *Pc = &q[0].vals[8]; // indexed from 0 to 7

	Cplx *Q = &q[0].vals[16 - 1]; // subtract 1 for indexing 1 to 7
	Cplx *Qc = &q[0].vals[23 - 1]; // subtract 1 for indexing 1 to 7


	memset(eval[0].vals, 0, sizeof(Matr<nV, nV+1>));

#pragma unroll
	for (int j = 1; j <= 7; ++j){

		/* eqnsI derivatives */

		/* dI/dA */
		eval[0].vals[(j - 1) * 22 + 0] = cplxAdd(cplxAdd(cplxSub(cplxSub(cplxMul({ -1, 0 }, cplxMul(Ac, Z0) \
			), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j]), Z0 \
			)), cplxMul(cplxMul(Q[j], Z0), cplxAdd(cplxMul({ -1, 0 }, Ac), cplxMul(\
			Pc[j], Z0)))), cplxMul(Z0, cplxAdd(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc \
			), cplxMul(Pc[j], Z0)))), cplxDiv(cplxMul(cplxMul(cplxAdd({ -1, 0 }, \
			Q[j]), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), U[j]));

		/* dI/dB */
		eval[0].vals[(j - 1) * 22 + 1] = { 0, 0 };

		/* dI/dC */
		eval[0].vals[(j - 1) * 22 + 2] = cplxSub(cplxAdd(cplxSub(cplxMul(Ac, Z0), cplxMul(Hc, Z0 \
			)), cplxMul(cplxMul(Q[j], Z0), cplxAdd(cplxMul({ -1, 0 }, Ac), cplxMul(\
			Pc[j], Z0)))), cplxDiv(\
			cplxMul(cplxMul(Q[j], cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), U[j]));

		/* dI/dD */
		eval[0].vals[(j - 1) * 22 + 3] = { 0, 0 };

		/* dI/dF */
		eval[0].vals[(j - 1) * 22 + 4] = { 0, 0 };

		/* dI/dG */
		eval[0].vals[(j - 1) * 22 + 5] = { 0, 0 };

		/* dI/dH */
		eval[0].vals[(j - 1) * 22 + 6] = cplxAdd(cplxMul({ -1, 0 }, cplxMul(U[j], cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(Pc[j], Z0))) \
			), cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, Cc), cplxMul(Pc[0], Z0))));

		/* dI/dAc */
		eval[0].vals[(j - 1) * 22 + 7] = cplxAdd(cplxAdd(cplxSub(cplxSub(cplxMul({ -1, 0 }, cplxMul(A, Z0) \
			), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j]), Z0 \
			)), cplxMul(cplxMul(Qc[j], Z0), cplxAdd(cplxMul({ -1, 0 }, A), cplxMul(\
			P[j], Z0)))), cplxMul(Z0, cplxAdd(cplxAdd(cplxMul({ -1, 0 }, A), C \
			), cplxMul(P[j], Z0)))), cplxMul(cplxMul(cplxAdd({ -1, 0 }, \
			Qc[j]), U[j]), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(P[0], Z0))));

		/* dI/dBc */
		eval[0].vals[(j - 1) * 22 + 8] = { 0, 0 };

		/* dI/dCc */
		eval[0].vals[(j - 1) * 22 + 9] = cplxSub(cplxAdd(cplxSub(cplxMul(A, Z0), cplxMul(H, Z0 \
			)), cplxMul(cplxMul(Qc[j], Z0), cplxAdd(cplxMul({ -1, 0 }, A), cplxMul(\
			P[j], Z0)))), cplxMul(cplxMul(Qc[j], U[j]), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))));

		/* dI/dDc */
		eval[0].vals[(j - 1) * 22 + 10] = { 0, 0 };

		/* dI/dFc */
		eval[0].vals[(j - 1) * 22 + 11] = { 0, 0 };

		/* dI/dGc */
		eval[0].vals[(j - 1) * 22 + 12] = { 0, 0 };

		/* dI/dHc */
		eval[0].vals[(j - 1) * 22 + 13] = cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(P[j], Z0))), U[j]) \
			), cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, C), cplxMul(P[0], Z0))));

		/* dI/dU */
		eval[0].vals[(j - 1) * 22 + 13 + j] = cplxSub(cplxMul(cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0)), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(Pc[j], Z0))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), cplxPow(U[j], { 2, 0 })));

		/* dI/dZ0 */
		eval[0].vals[j * 22 - 1] = cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(\
			cplxAdd(cplxAdd(cplxAdd(cplxSub(cplxAdd(cplxAdd(cplxAdd(cplxAdd(\
			cplxAdd(cplxMul(cplxMul(Ac, P[j]), Z0), cplxMul(cplxMul(Hc, P[0]), Z0 \
			)), cplxMul(cplxMul(A, Pc[j]), Z0)), cplxMul(cplxMul(H, Pc[0]), Z0 \
			)), cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), \
			C), Pc[j]), Q[j]), Z0)), cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac \
			), Cc), P[j]), Qc[j]), Z0)), cplxMul(cplxMul({ 3, 0 }, cplxAdd(cplxMul(P[j], Pc[j] \
			), cplxMul(P[0], Pc[0]))), cplxPow(Z0, { 2, 0 }))), cplxMul(cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j]), cplxAdd(cplxMul({ -1, 0 }, A \
			), cplxMul(P[j], Z0)))), cplxMul(Ac, cplxAdd(cplxAdd(cplxMul({ -1, 0 }, A), \
			C), cplxMul(P[j], Z0)))), cplxDiv(\
			cplxMul(cplxMul(Pc[0], cplxPow(Z0, { 2, 0 })), cplxAdd(cplxSub(cplxMul({ -1, 0 }, \
			A), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), U[j])), cplxMul(Hc, cplxAdd(cplxMul({ -1, 0 }, C), cplxMul(\
			P[0], Z0)))), cplxMul(cplxMul(Pc[j], U[j]), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0)))), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), \
			C), Q[j]), cplxAdd(cplxMul({ -1, 0 }, Ac), cplxMul(Pc[j], Z0)))), cplxMul(\
			A, cplxAdd(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), cplxMul(Pc[j], Z0)) \
			)), cplxMul(cplxMul(P[0], U[j]), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(Pc[j], Z0)) \
			)), cplxMul(H, cplxAdd(cplxMul({ -1, 0 }, Cc), cplxMul(Pc[0], Z0)) \
			)), cplxDiv(\
			cplxMul(cplxMul(P[j], cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), U[j])), cplxDiv(\
			cplxMul(cplxMul(cplxMul({ 2, 0 }, Z0), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(Pc[0], Z0))), U[j]));



		/* eqnsII derivatives */


		/* dII/dA */
		eval[0].vals[(7 + j - 1) * 22 + 0] = { 0, 0 };

		/* dII/dB */
		eval[0].vals[(7 + j - 1) * 22 + 1] = cplxSub(cplxAdd(cplxMul({ -1, 0 }, cplxMul(Fc, Z0)), cplxMul(\
			Pc[j], cplxPow(Z0, { 2, 0 }))), cplxDiv(\
			cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), U[j]));

		/* dII/dC */
		eval[0].vals[(7 + j - 1) * 22 + 2] = { 0, 0 };

		/* dII/dD */
		eval[0].vals[(7 + j - 1) * 22 + 3] = { 0, 0 };

		/* dII/dF */
		eval[0].vals[(7 + j - 1) * 22 + 4] = cplxAdd(cplxMul({ -1, 0 }, cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0)))), cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[0], Z0))));

		/* dII/dG */
		eval[0].vals[(7 + j - 1) * 22 + 5] = { 0, 0 };

		/* dII/dH */
		eval[0].vals[(7 + j - 1) * 22 + 6] = { 0, 0 };

		/* dII/dAc */
		eval[0].vals[(7 + j - 1) * 22 + 7] = { 0, 0 };

		/* dII/dBc */
		eval[0].vals[(7 + j - 1) * 22 + 8] = cplxSub(cplxAdd(cplxMul({ -1, 0 }, cplxMul(F, Z0)), cplxMul(\
			P[j], cplxPow(Z0, { 2, 0 }))), cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, F \
			), cplxMul(P[0], Z0))));

		/* dII/dCc */
		eval[0].vals[(7 + j - 1) * 22 + 9] = { 0, 0 };

		/* dII/dDc */
		eval[0].vals[(7 + j - 1) * 22 + 10] = { 0, 0 };

		/* dII/dFc */
		eval[0].vals[(7 + j - 1) * 22 + 11] = cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), U[j])), cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[0], Z0))));

		/* dII/dGc */
		eval[0].vals[(7 + j - 1) * 22 + 12] = { 0, 0 };

		/* dII/dHc */
		eval[0].vals[(7 + j - 1) * 22 + 13] = { 0, 0 };

		/* dII/dU */
		eval[0].vals[(7 + j - 1) * 22 + 13 + j] = cplxSub(cplxMul(cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(Pc[j], Z0))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxPow(U[j], { 2, 0 })));

		/* dII/dZ0 */
		eval[0].vals[(7 + j) * 22 - 1] = cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxAdd(cplxSub(\
			cplxAdd(cplxAdd(cplxAdd(cplxMul(cplxMul(cplxMul({ 2, 0 }, Bc), P[j]), Z0 \
			), cplxMul(cplxMul(Fc, P[0]), Z0)), cplxMul(cplxMul(cplxMul({ \
			2, 0 }, B), Pc[j]), Z0)), cplxMul(cplxMul(F, Pc[0]), Z0)), cplxMul(cplxMul({ \
			3, 0 }, cplxAdd(cplxMul(P[j], Pc[j]), cplxMul(P[0], Pc[0]))), cplxPow(Z0, { 2, 0 } \
			))), cplxDiv(\
			cplxMul(cplxMul(Pc[0], cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), U[j])), cplxMul(Fc, cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[0], Z0)))), cplxMul(cplxMul(Pc[j], U[j]), cplxAdd(cplxMul({ -1, 0 }, \
			F), cplxMul(P[0], Z0)))), cplxMul(cplxMul(\
			P[0], U[j]), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(Pc[j], Z0)))), cplxMul(\
			F, cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(Pc[0], Z0)))), cplxDiv(\
			cplxMul(cplxMul(P[j], cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, Fc \
			), cplxMul(Pc[0], Z0))), U[j])), cplxDiv(\
			cplxMul(cplxMul(cplxMul({ 2, 0 }, Z0), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(Pc[0], Z0))), U[j]));


		/* eqnsIII derivatives */


		/* dIII/dA */
		eval[0].vals[(14 + j - 1) * 22 + 0] = cplxAdd(cplxMul({ -1, 0 }, cplxMul(cplxMul(cplxSub(cplxAdd({ 1, 0 } \
			, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd({ -1, 0 }, \
			Q[j])), cplxAdd(cplxMul({ -1, 0 }, C), H))), \
			Q[j]), Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxSub(cplxAdd({ 1, 0 }, cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd({ -1, 0 }, \
			Q[j])), cplxAdd(cplxMul({ -1, 0 }, C), H))), \
			Q[j]), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j]));

		/* dIII/dB */
		eval[0].vals[(14 + j - 1) * 22 + 1] = cplxSub(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxMul(Z0, cplxAdd(cplxSub(cplxAdd({ -1, 0 }, cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B), F) \
			)), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), F), { 2, 0 }))), cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(P[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, B \
			), F)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(cplxMul(\
			U[j], cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), F), { 2, 0 }))), cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(P[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, \
			B), F)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd({ -1, 0 }, \
			cplxDiv(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), F))), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), F), { 2, 0 }))), cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(P[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, B \
			), F)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j] \
			)), cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), F), { 2, 0 }))), cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(P[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, \
			B), F)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

		/* dIII/dC */
		eval[0].vals[(14 + j - 1) * 22 + 2] = cplxSub(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxMul(Z0, cplxSub(cplxAdd(cplxSub(Q[j], cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), Q[j]), cplxAdd(cplxMul({ -1, 0 }, \
			C), H))), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), H), { 2, 0 }))), cplxDiv(\
			cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			A), C), Q[j])), cplxMul(P[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(cplxMul(\
			U[j], cplxSub(cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), H), { 2, 0 })), cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(P[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, \
			C), H)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxSub(cplxAdd(cplxSub(Q[j], cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), Q[j]), cplxAdd(cplxMul({ -1, 0 }, \
			C), H))), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), H), { 2, 0 }))), cplxDiv(\
			cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			A), C), Q[j])), cplxMul(P[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j] \
			)), cplxMul(cplxMul(\
			Z0, cplxSub(cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), H), { 2, 0 })), cplxDiv(\
			cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(P[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, \
			C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

		/* dIII/dD */
		eval[0].vals[(14 + j - 1) * 22 + 3] = cplxAdd(cplxSub(cplxSub(cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Dc), Gc), Z0)), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), F), U[j]))), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxMul({ -1, 0 }, B), F)));

		/* dIII/dF */
		eval[0].vals[(14 + j - 1) * 22 + 4] = cplxSub(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), Z0), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), F), { 2, 0 })) \
			), cplxMul(cplxMul(U[j], cplxAdd(cplxDiv(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, B), F)), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, F \
			), cplxMul(P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), \
			F), { 2, 0 })))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxMul(cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), F), { 2, 0 }), U[j]) \
			)), cplxMul(cplxMul(Z0, cplxAdd(cplxDiv(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, B), F)), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, F \
			), cplxMul(P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, B), \
			F), { 2, 0 })))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

		/* dIII/dG */
		eval[0].vals[(14 + j - 1) * 22 + 5] = cplxSub(cplxAdd(cplxAdd(cplxSub(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Dc \
			), Gc), Z0), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxMul({ -1, 0 }, C), H))), cplxDiv(\
			cplxMul(cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxMul({ -1, 0 }, C), H))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), H), U[j]))), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxMul({ -1, 0 }, C), H)));

		/* dIII/dH */
		eval[0].vals[(14 + j - 1) * 22 + 6] = cplxSub(cplxSub(cplxAdd(cplxDiv(cplxMul(cplxMul(cplxMul(cplxAdd(\
			cplxMul({ -1, 0 }, C), G), Z0), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), H), { 2, 0 } \
			)), cplxMul(cplxMul(U[j], cplxSub(cplxMul({ -1, 0 }, \
			cplxDiv(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, C \
			), H))), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), \
			H), { 2, 0 })))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxMul(cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), H), { 2, 0 }), U[j]) \
			)), cplxMul(cplxMul(Z0, cplxSub(cplxMul({ -1, 0 }, \
			cplxDiv(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, C \
			), H))), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, C), \
			H), { 2, 0 })))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

		/* dIII/dAc */
		eval[0].vals[(14 + j - 1) * 22 + 7] = cplxAdd(cplxMul({ -1, 0 }, cplxMul(cplxMul(cplxSub(cplxAdd({ 1, 0 } \
			, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd({ -1, 0 }, \
			Qc[j])), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), \
			Qc[j]), Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H))))), cplxMul(cplxMul(\
			cplxSub(cplxAdd({ 1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc \
			), Gc), cplxAdd({ -1, 0 }, Qc[j])), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), \
			Qc[j]), U[j]), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))));

		/* dIII/dBc */
		eval[0].vals[(14 + j - 1) * 22 + 8] = cplxSub(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxMul(Z0, cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxSub(cplxAdd({ -1, 0 }, cplxDiv(cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), { 2, 0 } \
			))), cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))))), cplxMul(cplxMul(\
			U[j], cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxSub(cplxAdd({ -1, 0 }, cplxDiv(cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), { 2, 0 } \
			))), cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd(cplxSub(A, \
			B), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), { 2, 0 })) \
			), cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc)))), U[j])), cplxMul(cplxMul(\
			Z0, cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), { 2, 0 })) \
			), cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc)))));

		/* dIII/dCc */
		eval[0].vals[(14 + j - 1) * 22 + 9] = cplxSub(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxMul(Z0, cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxSub(cplxAdd(cplxSub(Qc[j], cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Qc[j]), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 } \
			))), cplxDiv(cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(cplxMul(\
			U[j], cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxSub(cplxAdd(cplxSub(Qc[j], cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Qc[j]), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 } \
			))), cplxDiv(cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0)), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd(cplxSub(A, \
			B), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxSub(cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 } \
			)), cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j])), cplxMul(cplxMul(\
			Z0, cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxSub(cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 } \
			)), cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0)), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

		/* dIII/dDc */
		eval[0].vals[(14 + j - 1) * 22 + 10] = cplxAdd(cplxSub(cplxSub(cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, D), G), Z0)), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Fc))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, Fc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), U[j]))), cplxDiv(\
			cplxMul(cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc)));

		/* dIII/dFc */
		eval[0].vals[(14 + j - 1) * 22 + 11] = cplxSub(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), { 2, 0 }))), cplxDiv(\
			cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), U[j]), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), Fc), { 2, 0 }))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd(cplxSub(A, \
			B), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc)), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Fc \
			), cplxMul(Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Fc), { 2, 0 })))), U[j])), cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc)), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Fc \
			), cplxMul(Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Fc), { 2, 0 })))));

		/* dIII/dGc */
		eval[0].vals[(14 + j - 1) * 22 + 12] = cplxSub(cplxAdd(cplxAdd(cplxSub(cplxMul(cplxAdd(cplxMul({ -1, 0 }, D), \
			G), Z0), cplxDiv(cplxMul(cplxMul(Z0, cplxAdd(cplxSub(cplxMul({ -1, 0 }, \
			Ac), cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			Cc), Hc))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), U[j]))), cplxDiv(\
			cplxMul(cplxMul(U[j], cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), cplxDiv(\
			cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)));

		/* dIII/dHc */
		eval[0].vals[(14 + j - 1) * 22 + 13] = cplxSub(cplxAdd(cplxSub(cplxDiv(cplxMul(cplxMul(cplxMul(cplxAdd(\
			cplxMul({ -1, 0 }, Cc), Gc), Z0), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 })), cplxDiv(\
			cplxMul(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), U[j]), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 }))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd(cplxSub(A, \
			B), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxSub(cplxMul({ -1, 0 }, \
			cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, \
			Cc), Hc))), cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), Hc), { 2, 0 })))), U[j] \
			)), cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H)))), cplxSub(cplxMul({ -1, 0 }, cplxDiv(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxPow(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc), { 2, 0 })))));

		/* dIII/dU */
		eval[0].vals[(14 + j - 1) * 22 + 13 + j] = cplxSub(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxSub(cplxAdd(cplxSub(A, \
			B), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxPow(U[j], { 2, 0 })));

		/* dIII/dZ0 */
		eval[0].vals[(14 + j) * 22 - 1] = cplxSub(cplxAdd(cplxAdd(cplxSub(cplxSub(cplxSub(cplxAdd(cplxSub(\
			cplxAdd(cplxAdd(cplxSub(cplxMul(cplxAdd(cplxMul({ -1, 0 }, D), \
			G), cplxAdd(cplxMul({ -1, 0 }, Dc), Gc)), cplxMul(cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Pc[j]), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Pc[j]), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc))), Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H))))), cplxDiv(\
			cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Pc[0]), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Pc[0]), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc))), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), U[j])), cplxMul(cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Pc[j]), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Pc[j]), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc))), U[j]), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H))) \
			)), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Pc[0]), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Pc[0]), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc))), Z0), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H))) \
			)), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), P[0]), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), P[0]), cplxAdd(cplxMul({ -1, 0 }, \
			C), H))), U[j]), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), P[j]), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), P[j]), cplxAdd(cplxMul({ -1, 0 }, \
			C), H))), Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(\
			cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxMul(cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), P[0]), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), P[0]), cplxAdd(cplxMul({ -1, 0 }, \
			C), H))), Z0), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), P[j]), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), P[j]), cplxAdd(cplxMul({ -1, 0 }, \
			C), H))), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j] \
			)), cplxDiv(\
			cplxMul(cplxMul(cplxMul({ 2, 0 }, Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(A, \
			B), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j] \
			)), cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), \
			H))), cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, \
			Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));


	}

}


__device__ void JPAR(Vect<nV + 1> *Z, Vect<nP> *q, Matr<nV, nP> *eval){

	Cplx A = Z[0].vals[0];
	Cplx B = Z[0].vals[1];
	Cplx C = Z[0].vals[2];
	Cplx D = Z[0].vals[3];
	Cplx F = Z[0].vals[4];
	Cplx G = Z[0].vals[5];
	Cplx H = Z[0].vals[6];
	Cplx Ac = Z[0].vals[7];
	Cplx Bc = Z[0].vals[8];
	Cplx Cc = Z[0].vals[9];
	Cplx Dc = Z[0].vals[10];
	Cplx Fc = Z[0].vals[11];
	Cplx Gc = Z[0].vals[12];
	Cplx Hc = Z[0].vals[13];

	Cplx *U = &Z[0].vals[14 - 1]; // subtract 1 for indexing 1 to 7

	Cplx Z0 = Z[0].vals[21];

	Cplx *P = &q[0].vals[0]; // indexed from 0 to 7
	Cplx *Pc = &q[0].vals[8]; // indexed from 0 to 7

	Cplx *Q = &q[0].vals[16 - 1]; // subtract 1 for indexing 1 to 7
	Cplx *Qc = &q[0].vals[23 - 1]; // subtract 1 for indexing 1 to 7

	memset(eval[0].vals, 0, sizeof(Matr<nV, nP>));

#pragma unroll
	for (int j = 1; j <= 7; ++j){

		/* eqnsI derivatives */

		/* dI/dP0 */
		eval[0].vals[(j - 1) * 30] = cplxAdd(cplxSub(cplxMul(Hc, cplxPow(Z0, { 2, 0 })), cplxMul(\
			Pc[0], cplxPow(Z0, { 3, 0 }))), cplxMul(cplxMul(\
			U[j], Z0), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(Pc[j], Z0))));

		/* dI/dP */
		eval[0].vals[(j - 1) * 30 + j] = cplxAdd(cplxSub(cplxAdd(cplxMul(Ac, cplxPow(Z0, { 2, 0 } \
			)), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), \
			Cc), Qc[j]), cplxPow(Z0, { 2, 0 }))), cplxMul(Pc[j], cplxPow(Z0, { 3, 0 } \
			))), cplxDiv(cplxMul(cplxPow(Z0, { 3, 0 }), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), U[j]));

		/* dI/dPc0 */
		eval[0].vals[(j - 1) * 30 + 8] = cplxAdd(cplxSub(cplxMul(H, cplxPow(Z0, { 2, 0 })), cplxMul(\
			P[0], cplxPow(Z0, { 3, 0 }))), cplxDiv(\
			cplxMul(cplxPow(Z0, { 3, 0 }), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A \
			), cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(P[j], Z0))), U[j]));

		/* dI/dPc */
		eval[0].vals[(j - 1) * 30 + 8 + j] = cplxAdd(cplxSub(cplxAdd(cplxMul(A, cplxPow(Z0, { 2, 0 } \
			)), cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), \
			C), Q[j]), cplxPow(Z0, { 2, 0 }))), cplxMul(P[j], cplxPow(Z0, { 3, 0 } \
			))), cplxMul(cplxMul(U[j], Z0), cplxAdd(cplxMul({ -1, 0 }, H), cplxMul(\
			P[0], Z0))));

		/* dI/dQ */
		eval[0].vals[(j - 1) * 30 + 15 + j] = cplxAdd(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, A), \
			C), Z0), cplxAdd(cplxMul({ -1, 0 }, Ac), cplxMul(Pc[j], Z0))), cplxDiv(\
			cplxMul(cplxMul(cplxSub(A, \
			C), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, Hc), cplxMul(\
			Pc[0], Z0))), U[j]));

		/* dI/dQc */
		eval[0].vals[(j - 1) * 30 + 22 + j] = cplxAdd(cplxMul(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Ac), \
			Cc), Z0), cplxAdd(cplxMul({ -1, 0 }, A), cplxMul(P[j], Z0)) \
			), cplxMul(cplxMul(cplxSub(Ac, Cc), U[j]), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))));



		/* eqnsII derivatives */

		/* dII/dP0 */
		eval[0].vals[(7 + j - 1) * 30] = cplxAdd(cplxSub(cplxMul(Fc, cplxPow(Z0, { 2, 0 })), cplxMul(\
			Pc[0], cplxPow(Z0, { 3, 0 }))), cplxMul(cplxMul(\
			U[j], Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), cplxMul(Pc[j], Z0))));

		/* dII/dP */
		eval[0].vals[(7 + j - 1) * 30 + j] = cplxAdd(cplxSub(cplxMul(Bc, cplxPow(Z0, { 2, 0 })), cplxMul(\
			Pc[j], cplxPow(Z0, { 3, 0 }))), cplxDiv(\
			cplxMul(cplxPow(Z0, { 3, 0 }), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), U[j]));

		/* dII/dPc0 */
		eval[0].vals[(7 + j - 1) * 30 + 8] = cplxAdd(cplxSub(cplxMul(F, cplxPow(Z0, { 2, 0 })), cplxMul(\
			P[0], cplxPow(Z0, { 3, 0 }))), cplxDiv(\
			cplxMul(cplxPow(Z0, { 3, 0 }), cplxAdd(cplxMul({ -1, 0 }, B), cplxMul(\
			P[j], Z0))), U[j]));

		/* dII/dPc */
		eval[0].vals[(7 + j - 1) * 30 + 8 + j] = cplxAdd(cplxSub(cplxMul(B, cplxPow(Z0, { 2, 0 })), cplxMul(\
			P[j], cplxPow(Z0, { 3, 0 }))), cplxMul(cplxMul(U[j], Z0), cplxAdd(cplxMul({ -1, 0 }, \
			F), cplxMul(P[0], Z0))));

		/* dII/dQ */
		eval[0].vals[(7 + j - 1) * 30 + 15 + j] = { 0, 0 };

		/* dII/dQc */
		eval[0].vals[(7 + j - 1) * 30 + 22 + j] = { 0, 0 };



		/* eqnsIII derivatives */

		/* dIII/dP0 */
		eval[0].vals[(14 + j - 1) * 30] = cplxSub(cplxMul(cplxMul(U[j], cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), Z0), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), Z0), cplxAdd(cplxMul({ -1, 0 }, \
			C), H)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), cplxMul(cplxMul(\
			Z0, cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B \
			), D), Z0), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), Z0), cplxAdd(cplxMul({ -1, 0 }, \
			C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))));

		/* dIII/dP */
		eval[0].vals[(14 + j - 1) * 30 + j] = cplxAdd(cplxMul({ -1, 0 }, cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), Z0), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), Z0), cplxAdd(cplxMul({ -1, 0 }, \
			C), H)))), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), Z0), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), Z0), cplxAdd(cplxMul({ -1, 0 }, \
			C), H)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j]));

		/* dIII/dPc0 */
		eval[0].vals[(14 + j - 1) * 30 + 8] = cplxSub(cplxDiv(cplxMul(cplxMul(cplxPow(Z0, { 2, 0 }), cplxAdd(cplxMul({ -1, \
			0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Z0), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))), U[j]), cplxMul(cplxMul(\
			Z0, cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc \
			), Dc), Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Z0), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))));

		/* dIII/dPc */
		eval[0].vals[(14 + j - 1) * 30 + 8 + j] = cplxAdd(cplxMul({ -1, 0 }, cplxMul(cplxMul(Z0, cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Z0), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc)))), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H))))), cplxMul(cplxMul(\
			U[j], cplxAdd(cplxMul({ -1, 0 }, cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc \
			), Dc), Z0), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), Z0), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))));

		/* dIII/dQ */
		eval[0].vals[(14 + j - 1) * 30 + 15 + j] = cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxMul(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, A), C), cplxDiv(\
			cplxMul(cplxSub(A, C), cplxAdd(cplxMul({ -1, 0 }, C), \
			G)), cplxAdd(cplxMul({ -1, 0 }, C), \
			H))), Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(Ac, Bc), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), Dc), cplxAdd(cplxMul({ -1, 0 }, Bc \
			), cplxMul(Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc), cplxAdd(cplxSub(cplxMul({ -1, 0 }, Ac), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), Qc[j])), cplxMul(\
			Pc[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))))), cplxDiv(\
			cplxMul(cplxMul(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, A), C), cplxDiv(\
			cplxMul(cplxSub(A, C), cplxAdd(cplxMul({ -1, 0 }, C), \
			G)), cplxAdd(cplxMul({ -1, 0 }, C), \
			H))), cplxPow(Z0, { 2, 0 })), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, Bc), \
			Dc), cplxAdd(cplxMul({ -1, 0 }, Fc), cplxMul(\
			Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Bc), Fc))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, Cc), Gc), cplxAdd(cplxMul({ -1, 0 }, Hc \
			), cplxMul(Pc[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc)))), U[j]));

		/* dIII/dQc */
		eval[0].vals[(14 + j - 1) * 30 + 22 + j] = cplxAdd(cplxMul({ -1, 0 }, \
			cplxMul(cplxMul(cplxAdd(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), cplxDiv(\
			cplxMul(cplxSub(Ac, Cc), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc)), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Hc))), Z0), cplxAdd(cplxSub(cplxAdd(cplxSub(A, B), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), D), cplxAdd(cplxMul({ -1, 0 }, B \
			), cplxMul(P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), \
			G), cplxAdd(cplxSub(cplxMul({ -1, 0 }, A), cplxMul(\
			cplxAdd(cplxMul({ -1, 0 }, A), C), Q[j])), cplxMul(\
			P[j], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H))))), cplxMul(cplxMul(\
			cplxAdd(cplxAdd(cplxMul({ -1, 0 }, Ac), Cc), cplxDiv(\
			cplxMul(cplxSub(Ac, Cc), cplxAdd(cplxMul({ -1, 0 }, Cc), \
			Gc)), cplxAdd(cplxMul({ -1, 0 }, Cc), Hc))), U[j]), cplxAdd(cplxMul({ -1, 0 }, \
			cplxDiv(cplxMul(cplxAdd(cplxMul({ -1, 0 }, B), \
			D), cplxAdd(cplxMul({ -1, 0 }, F), cplxMul(\
			P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, B), F))), cplxDiv(\
			cplxMul(cplxAdd(cplxMul({ -1, 0 }, C), G), cplxAdd(cplxMul({ -1, 0 }, H \
			), cplxMul(P[0], Z0))), cplxAdd(cplxMul({ -1, 0 }, C), H)))));


	}

}

