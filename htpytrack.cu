#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "htpysupport.cuh"
#include "rungekutta.cuh"

// Runge-Kutta settings
#define sinit 1. // Initial step size
#define smax 100. // Maximum step size
#define smin 0.000001 // Minimum step size
#define RKtol 0.001 // Runge-Kutta tolerance
#define cautionSteps 2 // After a step rejection, stepsize is not allowed to increase again until after a few successful steps

// Other settings
#define NewtIts 10 // Max number of Newton iterations during the corrector step
#define NewtTol 0.00000001 // Desired accuracy for constraint satisfaction during the Newton corrector step
#define NewtResidual 1  // 0 to use norm(H), 1 to use norm(Ydiff)
#define NewtAttempts 3 // Number of times to try Newton's method with different patches
#define SharpIts 30 // Max number of Newton iterations for final sharpening
#define SharpTol 0.000000000001 // Desired accuracy for constraint satisfaction for final sharpening
#define SharpResidual 1 // 0 to use norm(H), 1 to use norm(Ydiff)
#define SharpAttempts 3 // Number of times to try sharpening with different patches
#define SharpTrigger 0.00001 // How close to t = 1 we must be to trigger final sharpening
#define NudgeToOneTrigger 0.00001 // If t is very close to 1, push it a to one
#define maxSteps 2000 // max number of steps, counting each advancement of t
#define maxMinisteps 3000 // max number of ministeps, counting every call to either RK or Newton functions

// Constants
#define UpperRand 2. // upper limit of randomly generated numbers
#define LowerRand -2. // lower limit of randomly generated numbers

__device__ void TrackPath(Vect<nV> *spt, Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma, 
	curandState randState, Vect<nV + 2> *Yfinal, int *errorCode){

	Vect<nV + 2> space[4];
	Vect<nV + 2> *Vc = &space[0];
	Vect<nV + 2> *Yn = &space[1];
	Vect<nV + 2> *Vn = &space[2];
	Vect<nV + 2> *YN = &space[3];


	/*----------Initialization----------*/

	// Initialize the tracked root Yc
	Vect<nV + 2> *Yc = Yfinal;
#pragma unroll
	for (int i = 0; i < nV; ++i){ Yc[0].vals[i] = spt[0].vals[i]; } // copy in start point values
	Yc[0].vals[nV] = { 1, 0 }; // set homogeneous coordinate to 1
	Yc[0].vals[nV + 1] = { 0, 0 }; // set t=0


	Vect<nV + 2> u; // assign random patch values
#pragma unroll 1
	for (int i = 0; i < nV + 2; ++i){
		u.vals[i].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
		u.vals[i].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	}

	SwitchPatch(&u, Yc, Yc); // adjust Yc for the new patch

	// assign a random velocity patch for initially computing Vc, the result is the same for any patch
#pragma unroll 1
	for (int i = 0; i < nV + 2; ++i){
		Vc[0].vals[i].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
		Vc[0].vals[i].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
	}

	IVP(false, Yc, Vc, &u, spm, fpm, gamma, Vc); // compute initial velocity

	Real s, err, scale, residual;
	s = sinit; // initial step size
	bool tstepQ = false; // track via arclength, not t
	int streak = cautionSteps - 1; // streak of successful steps
	int step = 0; // step number
	int ministep = 0;
	*errorCode = 0;





	while (true){ // Begin main loop

		if (step == maxSteps){ *errorCode = 1; return; }

		while (true){ // Begin correction loop

			while (true){ // Begin final steps loop

				while (true){ // Begin predictor loop

					// Perform Runge-Kutta prediction
					DormandPrince65(s, RKtol, tstepQ, Yc, Vc, &u, spm, fpm, gamma, Yn, Vn, &err, &scale);
					if (ministep++ == maxMinisteps){ *errorCode = 2; return; } // to prevent non-exiting loop

					if (err <= RKtol){ break; } // If within tolerance, the prediction is accepted

					// If it was unsuccessful, check for error condition
					if (s <= smin){ *errorCode = 3; return; }

					s *= scale;// If it was unsuccessful and there is no error condition, adjust the step size and try again
					if (s < smin){ s = smin*0.9; }

					streak = 0; // Mark the step rejection by setting the success streak to zero

				} // End predictor loop






				// If t is less than 0.99..., continue to Newton iterations
				if (Yn[0].vals[nV + 1].x <= 1. - NudgeToOneTrigger){ break; }

				// If t is very close to 1, head to sharpening
				if (abs(1. - Yn[0].vals[nV + 1].x) < SharpTrigger){ break; }

				// Set up a final prediction step to land on t = 1
				tstepQ = true;
				//s = 1 - Yc[0].vals[nV + 1].x - SharpTrigger*0.1;
				s = 1 - Yc[0].vals[nV + 1].x;

			} // End final steps loop






			// If t is very close to 1, head to sharpening
			if (abs(1. - Yn[0].vals[nV + 1].x) < SharpTrigger){ break; }

			int i = 0; // Begin Newton patch switching loop
			while (i < NewtAttempts){

				// Perform Newton's method
				*YN = *Yn; //  set Newton start point
				Newton(Yn[0].vals[nV + 1].x, Vn, &u, spm, fpm, gamma, NewtIts, NewtTol, NewtResidual, YN, &residual);
				if (ministep++ == maxMinisteps){ *errorCode = 2; return; } // to prevent non-exiting loop

				if (residual < NewtTol){ break; } // If Newton's method was successful, move on

				// If not, generate a new projective patch, then try again
#pragma unroll 1
				for (int j = 0; j < nV + 2; ++j){
					u.vals[j].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
					u.vals[j].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
				}
				SwitchPatch(&u, Yn, Yn); // update Yn for new patch

				i++;
			} // End Newton patch switching loop

			if (residual < NewtTol){ break; }  // If Newton's method was successful, move on

			// If it was unsuccessful, check for error condition
			if (s <= smin){ *errorCode = 4; return; }

			s *= 0.5; // If it was unsuccessful and there is no error condition, then halve the stepsize and try again
			if (s < smin) { s = smin*0.9; }

			streak = 0; // Mark the step rejection by setting the success streak to zero

		} // End correction loop






		// If t is very close to 1, head to sharpening
		if (abs(1. - Yn[0].vals[nV + 1].x) < SharpTrigger){ break; }

		*Yc = *YN; // update position
		*Vc = *Vn; // update velocity
		streak++; // increment the streak of successes
		if (streak < cautionSteps && scale > 1.){ scale = 1.; } // limit step scale if the streak of successes is too low
		s *= scale; // update step size
		if (s > smax){ s = smax; } // step size must obey limits
		else if (s < smin){ s = smin*0.9; }
		step++; // increment homotopy step number

	} // End main loop






	// Begin Sharpening

	int i = 0;
	while (i < SharpAttempts){ // Begin sharpening patch switching loop

		// Perform final sharpening
		*Yfinal = *Yn;
		Newton(1., Vn, &u, spm, fpm, gamma, SharpIts, SharpTol, SharpResidual, Yfinal, &residual); // consider normalizing Vn
		if (ministep++ == maxMinisteps){ *errorCode = 2; return; } // to prevent non-exiting loop

		if (residual < SharpTol){ break; } // Check if sharpening was successful

		// If not, generate a new projective patch, then try again
#pragma unroll 1
		for (int j = 0; j < nV + 2; ++j){
			u.vals[j].x = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
			u.vals[j].y = (UpperRand - LowerRand)*curand_uniform_double(&randState) + LowerRand;
		}
		SwitchPatch(&u, Yn, Yn); // update Yn for new patch

		i++;
	} // End sharpening patch switching loop

	// Check if sharpening was successful
	if (residual >= SharpTol){ *errorCode = 5; return; }



}