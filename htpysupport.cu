#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include "support.cuh"
#include "htpykinematics.cuh"



__device__ void SwitchPatch(Vect<nV + 2> *unew, Vect<nV + 2> *Yold, Vect<nV + 2> *Ynew){

	// Switch to a new projective patch
	// unew: new projective patch
	// Yold: old coordinates
	// Ynew: (output) new coordinates

	Cplx *ut = &unew[0].vals[nV+1];
	Cplx *t = &Yold[0].vals[nV+1];


	Cplx denominator = { 0, 0 };
#pragma unroll
	for (int i = 0; i < nV + 1; ++i){
		denominator = cplxAdd(denominator, cplxMul(unew[0].vals[i], Yold[0].vals[i]));
	}

	Cplx scalar = cplxDiv(cplxSub({ 1, 0 }, cplxMul(*ut, *t)), denominator);
	
#pragma unroll
	for (int i = 0; i < nV + 1; ++i){
		Ynew[0].vals[i] = cplxMul(scalar, Yold[0].vals[i]);
	}
	Ynew[0].vals[nV + 1] = *t;

}


__device__ void q(Real t, Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, Vect<nP> *params){

#pragma unroll
	for (int i = 0; i < nP; ++i){

		params[0].vals[i] = cplxDiv(
			cplxAdd(cplxMul(cplxMul(*gamma, { 1. - t, 0 }), spm[0].vals[i]), cplxMul({ t, 0 }, fpm[0].vals[i])),
			cplxAdd(cplxMul(*gamma, { 1. - t, 0 }), { t, 0 })
			);

	}
}


__device__ void dqdt(Real t, Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, Vect<nP> *dparams){

#pragma unroll
	for (int i = 0; i < nP; ++i){

		dparams[0].vals[i] = cplxDiv(
			cplxMul(cplxSub(fpm[0].vals[i],spm[0].vals[i]), *gamma),
			cplxPow( cplxAdd(cplxMul(*gamma, { 1. - t, 0 }), { t, 0 }), 2)
			);

	}
}

__device__ void H(Vect<nV + 2> *Y, Vect<nV + 2> *u, 
	Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, Vect<nV+1> *eval){

	Vect<nV + 1> *Z = eval;
#pragma unroll
	for (int i = 0; i < nV+1; ++i){ Z[0].vals[i] = Y[0].vals[i]; }

	Real t = Y[0].vals[nV + 1].x;

	Vect<nP> params;
	q(t, spm, fpm, gamma, &params);

	Vect<nV> *Feval = (Vect<nV>*)eval;
	FSYS(Z, &params, Feval);

#pragma unroll
	for (int i = 0; i < nV; ++i){
		eval[0].vals[i] = Feval[0].vals[i];
	}

	eval[0].vals[nV] = cplxSub(u[0].dot(Y[0]), { 1, 0 });

}


__device__ void JH(Vect<nV + 2> *Y, Vect<nV + 2> *u,
	Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, Matr<nV+1,nV+2> *eval){

	// Store Z, A, & B in output memory space
	
	Vect<nV + 1> *Z = (Vect<nV + 1>*)eval;
#pragma unroll
	for (int i = 0; i < nV + 1; ++i){ Z[0].vals[i] = Y[0].vals[i]; }

	Matr<nV, nV + 1> *A = (Matr<nV, nV + 1>*)&eval[0].vals[nV + 1];

	Vect<nV> *B = (Vect<nV>*)&eval[0].vals[nV + 1 + (nV)*(nV + 1)];


	// Compute dF/dZ

	Real t = Y[0].vals[nV + 1].x;

	Vect<nP> params;
	q(t, spm, fpm, gamma, &params);

	JVAR(Z, &params, A);


	// Compute dF/dt

	Matr<nV, nP> dFdq;
	JPAR(Z, &params, &dFdq);

	Vect<nP> *dparams = &params;
	dqdt(t, spm, fpm, gamma, dparams);

	*B = dFdq.mul(*dparams);


	// Assemble Jacobian

#pragma unroll
	for (int i = 0; i < nV; ++i){
#pragma unroll 1
		for (int j = 0; j < nV + 1; ++j){
			eval[0].vals[i*(nV + 2) + j] = A[0].vals[i*(nV + 1) + j];
		}
	}

#pragma unroll
	for (int i = 0; i < nV; ++i){
		eval[0].vals[i*(nV + 2) + nV + 1] = B[0].vals[i];
	}

#pragma unroll
	for (int i = 0; i < nV + 2; ++i){
		eval[0].vals[(nV)*(nV + 2) + i] = u[0].vals[i];
	}

}

__device__ void LinearSolve(Matr<nV + 2, nV + 2> *mat, Vect<nV + 2> *b, Vect<nV + 2> *x){

	const int n = nV + 2;

	//CAUTION: This function modifies the memory locations of argument mat and b

	Cplx *a[n]; // construct an array of pointers to the first element of each row
#pragma unroll
	for (int i = 0; i < n; ++i){ a[i] = &mat[0].vals[i*n]; }


	//int P[n];
	//for (int i = 0; i < n; ++i){ P[i] = i; }

#pragma unroll
	for (int i = 0; i < n; ++i){

		// Find maximum absolute value of the current pivot column
		Real val;
		Real maxAbs = 0.;
		int maxRow = i;
#pragma unroll //1
		for (int k = i; k < n; ++k){
			if ((val = a[k][i].x * a[k][i].x + a[k][i].y * a[k][i].y) > maxAbs){
				maxAbs = val;
				maxRow = k;
			}
		}

		if (maxAbs < 0.0000001){ /* the matrix might be singular, contemplate reporting it */ }

		// Swap rows
		if (maxRow != i){

			Cplx *ptr = a[i];
			a[i] = a[maxRow];
			a[maxRow] = ptr;

			//int index = P[i];
			//P[i] = P[maxRow];
			//P[maxRow] = index;

			Cplx *bvalue = &x[0].vals[0]; // share space
			*bvalue = b[0].vals[i];
			b[0].vals[i] = b[0].vals[maxRow];
			b[0].vals[maxRow] = *bvalue;

		}

		// Index through rows to perform elimination arithmetic
		for (int j = i + 1; j < n; ++j){

			//Cplx scalar = cplxDiv(a[j][i], a[i][i]);
			a[j][i] = cplxDiv(a[j][i], a[i][i]);
#pragma unroll //1
			for (int k = i + 1; k < n; ++k){
				a[j][k] = cplxSub(a[j][k], cplxMul(a[i][k], a[j][i]));
			}

			b[0].vals[j] = cplxSub(b[0].vals[j], cplxMul(b[0].vals[i], a[j][i]));

		}
		// a now contains the LU decomposition of which the diagonal belongs to U (the diagonal of L is all 1s)

	}

	// Back substitution
#pragma unroll
	for (int i = n - 1; i > -1; --i){

		Cplx *sum = &x[0].vals[i]; // share space
		*sum = { 0., 0. };
#pragma unroll //1
		for (int k = i + 1; k < n; ++k){
			*sum = cplxFma(a[i][k], x[0].vals[k], *sum);
		}

		x[0].vals[i] = cplxMul(cplxDiv({ 1, 0 }, a[i][i]), cplxSub(b[0].vals[i], *sum));

	}

	//for (int i = 0; i < n; ++i){

	//	x[0].vals[i] = {(double)(P[i]+1), 0.};

	//}
}




__device__ void IVP(bool tstepQ, Vect<nV + 2> *Yc, Vect<nV + 2> *Vc, Vect<nV + 2> *u,
	Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, Vect<nV + 2> *dYds){

	// Initial Value Problem for the Runge-Kutta predictor, computes velocity
	// tstepQ: false: dYds is returned as unit vector for scaling by arclength
	//         true:  dYds is returned with t component equal to 1 for scaling by t
	// Yc: current position
	// Vc: current velocity
	// u: projective patch
	// spm, fpm, gamma: start parameters, final parameters, gamma
	// dYds: (output) velocity

	// Compute JH
	Matr<nV + 2, nV + 2> JHaug;
	JH(Yc, u, spm, fpm, gamma, (Matr<nV + 1, nV + 2>*)&JHaug);

	// Append Vc patch
#pragma unroll
	for (int j = 0; j < nV + 2; ++j){
		JHaug.vals[(nV + 1)*(nV + 2) + j] = Vc[0].vals[j];
	}

	// Rescale the normal vectors that comprise JHaug
#pragma unroll
	for (int i = 0; i < nV + 1; ++i){

		Real maxVal = 0;
#pragma unroll //1
		for (int j = 0; j < nV + 2; ++j){
			if (abs(JHaug.vals[i*(nV + 2) + j].x) > maxVal){ maxVal = abs(JHaug.vals[i*(nV + 2) + j].x); }
			if (abs(JHaug.vals[i*(nV + 2) + j].y) > maxVal){ maxVal = abs(JHaug.vals[i*(nV + 2) + j].y); }
		}

		Real *scale = &maxVal;
		*scale = 1 / maxVal;
#pragma unroll //1
		for (int j = 0; j < nV + 2; ++j){
			JHaug.vals[i*(nV + 2) + j].x = *scale * JHaug.vals[i*(nV + 2) + j].x;
			JHaug.vals[i*(nV + 2) + j].y = *scale * JHaug.vals[i*(nV + 2) + j].y;
		}
	}

	Vect<nV + 2> b;
	memset(b.vals, 0, sizeof(Vect<nV+1>));
	b.vals[nV + 1] = { 1.0, 0 };

	LinearSolve(&JHaug, &b, dYds);

	Cplx scaleCplx;
	if (tstepQ == false){
		Real *theta = &scaleCplx.x;
		*theta = atan2(dYds[0].vals[nV + 1].y, dYds[0].vals[nV + 1].x);
		scaleCplx = cplxMul({ cos(*theta), -sin(*theta) }, { 1.0 / dYds[0].norm(), 0 });
	}
	else if (tstepQ == true) {
		scaleCplx = cplxDiv({ 1.0, 0 }, dYds[0].vals[nV + 1]);
	}

	dYds[0] = dYds[0].scale(scaleCplx);
	dYds[0].vals[nV + 1].y = 0.;


}


__device__ void LinearSolveNewton(Matr<nV + 2, nV + 2> *mat, Vect<nV + 1> *b, Real *tdiff, Vect<nV + 2> *x){

	// Special linear solver for Newton's method
	//mat: the augmented Jacobian JHaug
	//b: the nonaugmented version of -H (this is what makes it special)
	//tdiff: the change in t (the last element of Y)
	//x: Y

	const int n = nV + 2;

	//CAUTION: This function modifies the memory locations of argument mat and b

	Cplx *a[n]; // construct an array of pointers to the first element of each row
#pragma unroll
	for (int i = 0; i < n; ++i){ a[i] = &mat[0].vals[i*n]; }


	//int P[n];
	//for (int i = 0; i < n; ++i){ P[i] = i; }

#pragma unroll
	for (int i = 0; i < n; ++i){

		// Find maximum absolute value of the current pivot column
		Real val;
		Real maxAbs = 0.;
		int maxRow = i;
#pragma unroll //1
		for (int k = i; k < n - 1; ++k){ // Modification: the last row is not included
			if ((val = a[k][i].x * a[k][i].x + a[k][i].y * a[k][i].y) > maxAbs){
				maxAbs = val;
				maxRow = k;
			}
		}

		if (maxAbs < 0.0000001){ /* the matrix might be singular, contemplate reporting it */ }

		// Swap rows
		if (maxRow != i){

			Cplx *ptr = a[i];
			a[i] = a[maxRow];
			a[maxRow] = ptr;

			//int index = P[i];
			//P[i] = P[maxRow];
			//P[maxRow] = index;

			Cplx *bvalue = &x[0].vals[0]; // share space
			*bvalue = b[0].vals[i];
			b[0].vals[i] = b[0].vals[maxRow];
			b[0].vals[maxRow] = *bvalue;

		}

		// Index through rows to perform elimination arithmetic
		for (int j = i + 1; j < n - 1; ++j){ // Modification: the last row is not included

			//Cplx scalar = cplxDiv(a[j][i], a[i][i]);
			a[j][i] = cplxDiv(a[j][i], a[i][i]);
#pragma unroll //1
			for (int k = i + 1; k < n; ++k){
				a[j][k] = cplxSub(a[j][k], cplxMul(a[i][k], a[j][i]));
			}

			b[0].vals[j] = cplxSub(b[0].vals[j], cplxMul(b[0].vals[i], a[j][i]));

		}
		// a now contains the LU decomposition of which the diagonal belongs to U (the diagonal of L is all 1s)

	}

	x[0].vals[n - 1] = { *tdiff, 0. }; // Modification: define last element of x

	// Back substitution
#pragma unroll
	for (int i = n - 2; i > -1; --i){

		Cplx *sum = &x[0].vals[i]; // share space
		*sum = { 0., 0. };
#pragma unroll //1
		for (int k = i + 1; k < n; ++k){
			*sum = cplxFma(a[i][k], x[0].vals[k], *sum);
		}

		x[0].vals[i] = cplxMul(cplxDiv({ 1, 0 }, a[i][i]), cplxSub(b[0].vals[i], *sum));

	}



	//for (int i = 0; i < n; ++i){

	//	x[0].vals[i] = {(double)(P[i]+1), 0.};

	//}


}






__device__ void Newton(Real tgoal, Vect<nV + 2> *Vn, Vect<nV + 2> *u, 
	Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, 
	int maxIts, Real tol, int residualtype, Vect<nV+2> *YN, Real *residual){

	//Newton's method
	//tgoal: the value of t for which we intend to solve H
	//Vn: velocity computed from Runge-Kutta prediction, used to construct velocity patch
	//u: projective patch
	//spm, fpm, gamma: start parameters, final parameters, gamma
	//maxIts: maximum number of Newton iterations
	//tol: root is satisfactorily approximated when residual < tol
	//residualtype: 0 to use norm(H), 1 to use norm(Ydiff)
	//YN: (input/output) input starting point Ystart here, the function returns sharpened point YN
	//residual: (output) Newton residual


	// Initialization
	int i = 0;
	*residual = 100.0;

	while (i < maxIts && *residual > tol){


		// Compute JH
		Matr<nV + 2, nV + 2> JHaug;
		JH(YN, u, spm, fpm, gamma, (Matr<nV + 1, nV + 2>*)&JHaug);

		// Append Vc patch
#pragma unroll
		for (int j = 0; j < nV + 2; ++j){
			JHaug.vals[(nV + 1)*(nV + 2) + j] = Vn[0].vals[j];
		}

		// Compute H
		Vect<nV + 1> Heval;
		H(YN, u, spm, fpm, gamma, &Heval);


		// Scaling to improve linear solve
#pragma unroll
		for (int j = 0; j < nV + 1; ++j){

			// Find maximum value in each matrix row
			Real maxVal = 0;
#pragma unroll //1
			for (int k = 0; k < nV + 2; ++k){
				if (abs(JHaug.vals[j*(nV + 2) + k].x) > maxVal){ maxVal = abs(JHaug.vals[j*(nV + 2) + k].x); }
				if (abs(JHaug.vals[j*(nV + 2) + k].y) > maxVal){ maxVal = abs(JHaug.vals[j*(nV + 2) + k].y); }
			}

			Real *scale = &maxVal;
			*scale = 1 / maxVal;

			// Scale elements of H
			Heval.vals[j].x = *scale * Heval.vals[j].x;
			Heval.vals[j].y = *scale * Heval.vals[j].y;

			// Scale rows of JH
#pragma unroll //1
			for (int k = 0; k < nV + 2; ++k){
				JHaug.vals[j*(nV + 2) + k].x = *scale * JHaug.vals[j*(nV + 2) + k].x;
				JHaug.vals[j*(nV + 2) + k].y = *scale * JHaug.vals[j*(nV + 2) + k].y;
			}

		}

		// Compute Ydiff
		Vect<nV + 2> Ydiff;
		Real tdiff = tgoal - YN[0].vals[nV + 1].x;
		Vect<nV + 1> b = Heval.scale(-1.0);
		LinearSolveNewton(&JHaug, &b, &tdiff, &Ydiff);

		// Update YN
		YN[0] = YN[0].add(Ydiff);

		// Update residual
		if (residualtype == 0){
			H(YN, u, spm, fpm, gamma, &Heval);
			*residual = Heval.norm();
		}
		else{
			*residual = Ydiff.norm();
		}

		i++;
	}

}