#include "htpysupport.cuh"

__device__ void DormandPrince65(Real s, Real tol, bool tstepQ, Vect<nV + 2> *Yc, Vect<nV + 2> *Vc,
	Vect<nV + 2> *u, Vect<nP> *spm, Vect<nP> *fpm, Cplx *gamma,
	Vect<nV + 2> *Yn, Vect<nV + 2> *Vn, Real *err, Real *scale){

	// s: step size
	// tol: Runge-Kutta tolerance parameter
	// tstepQ: false: output vector scalable by arclength i.e. a unit vector
	//          true: output vector scalable by t
	// Yc: current position
	// Vc: current velocity
	// u: projective patch
	// spm, fpm, gamma: start parameters, final parameters, gamma
	// Yn: (output) predicted next position
	// Vn: (output) predicted next velocity
	// err: estimated error
	// scale: proposed scaling of step size

	// pointers for the necessary vectors
	Vect<nV + 2> *k1, *k2, *k3, *k4, *k5, *k6, *k7, *k8, *LTE, *Yarg;

	// save some space by sharing memory locations
	Yarg = Yn;
	LTE = Yn;
	k8 = Vn; // Vn is identically k8

	// create new space the rest of the vectors
	Vect<nV + 2> space[7];
	k1 = &space[0];
	k2 = &space[1];
	k3 = &space[2];
	k4 = &space[3];
	k5 = &space[4];
	k6 = &space[5];
	k7 = &space[6];


	// k1
	IVP(tstepQ, Yc, Vc, u, spm, fpm, gamma, k1);

	// k2
	*Yarg = Yc[0]
		.add(k1[0].scale(1./10. * s));
	IVP(tstepQ, Yarg, k1, u, spm, fpm, gamma, k2);

	// k3
	*Yarg = Yc[0]
		.add(k1[0].scale(-2. / 81. * s)
		.add(k2[0].scale(20. / 81. * s)));
	IVP(tstepQ, Yarg, k2, u, spm, fpm, gamma, k3);

	// k4
	*Yarg = Yc[0]
		.add(k1[0].scale(615. / 1372. * s)
		.add(k2[0].scale(-270. / 343. * s)
		.add(k3[0].scale(1053. / 1372. * s))));
	IVP(tstepQ, Yarg, k3, u, spm, fpm, gamma, k4);

	// k5
	*Yarg = Yc[0]
		.add(k1[0].scale(3243. / 5500. * s)
		.add(k2[0].scale(-54. / 55. * s)
		.add(k3[0].scale(50949. / 71500. * s)
		.add(k4[0].scale(4998. / 17875. * s)))));
	IVP(tstepQ, Yarg, k4, u, spm, fpm, gamma, k5);

	// k6
	*Yarg = Yc[0]
		.add(k1[0].scale(-26492. / 37125. * s)
		.add(k2[0].scale(72. / 55. * s)
		.add(k3[0].scale(2808. / 23375. * s)
		.add(k4[0].scale(-24206. / 37125. * s)
		.add(k5[0].scale(338. / 459. * s))))));
	IVP(tstepQ, Yarg, k5, u, spm, fpm, gamma, k6);

	// k7
	*Yarg = Yc[0]
		.add(k1[0].scale(5561. / 2376. * s)
		.add(k2[0].scale(-35. / 11. * s)
		.add(k3[0].scale(-24117. / 31603. * s)
		.add(k4[0].scale(899983. / 200772. * s)
		.add(k5[0].scale(-5225. / 1836. * s)
		.add(k6[0].scale(3925. / 4056. * s)))))));
	IVP(tstepQ, Yarg, k6, u, spm, fpm, gamma, k7);

	// k8
	*Yarg = Yc[0]
		.add(k1[0].scale(465467. / 266112. * s)
		.add(k2[0].scale(-2945. / 1232. * s)
		.add(k3[0].scale(-5610201. / 14158144. * s)
		.add(k4[0].scale(10513573. / 3212352. * s)
		.add(k5[0].scale(-424325. / 205632. * s)
		.add(k6[0].scale(376225. / 454272. * s)))))));
	IVP(tstepQ, Yarg, k7, u, spm, fpm, gamma, k8);


	// Compute error
	*LTE = k1[0].scale(-13. / 2400. * s)
		.add(k3[0].scale(19683. / 618800. * s)
		.add(k4[0].scale(-2401. / 31200. * s)
		.add(k5[0].scale(65. / 816. * s)
		.add(k6[0].scale(-15. / 416. * s)
		.add(k7[0].scale(-521. / 5600. * s)
		.add(k8[0].scale(1. / 10. * s)))))));

	*err = LTE[0].norm();

	if (*err > 10e-17){ *scale = 0.8*pow((tol / (*err)), 1. / 6.); }
	else{ *scale = 5.0; }

	if (*scale > 5){ *scale = 5; }

	// Compute prediction
	*Yn = Yc[0]
		.add(k1[0].scale(61. / 864. * s)
		.add(k3[0].scale(98415. / 321776. * s)
		.add(k4[0].scale(16807. / 146016. * s)
		.add(k5[0].scale(1375. / 7344. * s)
		.add(k6[0].scale(1375. / 5408. * s)
		.add(k7[0].scale(-37. / 1120. * s)
		.add(k8[0].scale(1. / 10. * s))))))));

}