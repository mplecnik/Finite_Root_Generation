#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>

#include "htpysupport.cuh"
#include "rungekutta.cuh"
#include "htpytrack.cuh"
#include "frgkinematics.cuh"
#include "htpykinematics.cuh"
#include "frgsupport.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <list>
#include <deque>
#include <chrono>

#define nBlocks 80 // blocks per GPU round
#define nThreads 256 // threads per block
#define nPaths 20480 // nBlocks*nThreads
//#define nRounds 0 // number of GPU rounds to execute

__global__ void InitRandStates(curandState *randStates, unsigned long long seed){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, i, 0, &randStates[i]);
}

__global__ void Homotopy(curandState *randStates, Vect<nP> *fpm, Vect<nV> *fpt, int *errorCodes, 
	Vect<nV + 2> *vectDebug, Matr<nV + 2, nV + 2> *matrDebug){
	
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	//const int tid = threadIdx.x;

	Cplx gamma = { 0.39231132891081666, 1.0062265257923928 };

	// Set up the dehomogenizing patch
	Vect<nV + 2> u_dehom;
	memset(u_dehom.vals, 0, sizeof(Vect<nV + 2>));
	(u_dehom.vals[nV]) = { 1., 0 };


	Vect<nV + 2> Yfinal;
	Vect<nV> *spt = (Vect<nV>*)&Yfinal; // spt shares space with Yfinal
	Vect<nP> spm;

	// Generate startpoint and start parameters
	GenerateStart(randStates[i], spt, &spm);

	// Track homotopy path
	TrackPath(spt, &spm, fpm, &gamma, randStates[i], &Yfinal, &errorCodes[i]);

	// Dehomogenize
	SwitchPatch(&u_dehom, &Yfinal, &Yfinal);

	// Compute cognates
	Vect<nV> cognSolns[2];
	cognSolns[0].set(Yfinal.vals);
	CognateSolns(&cognSolns[0], fpm, cognSolns);

	// Copy dehomogenized solution to output
	fpt[i] = cognSolns[0];



	//Vect<nV + 2> u;
	//u.vals[0] = { -0.7419102437270411, 0.1470056092936725 };
	//u.vals[1] = { -0.1184452756144854, 0.7849189775791263 };
	//u.vals[2] = { -1.410166698195684, 0.08896065720505586 };
	//u.vals[3] = { 0.2829955276217886, 1.838321754705803 };
	//u.vals[4] = { 0.2088677614358758, 1.790054555927655 };
	//u.vals[5] = { -0.4641029177321441, -1.364677741989282 };
	//u.vals[6] = { -0.8065399895290044, 1.666513959478181 };
	//u.vals[7] = { 1.004010993250414, -0.2958848402072620 };
	//u.vals[8] = { 0.6511473613113612, -0.6025404144105462 };
	//u.vals[9] = { -0.4863182555382233, 0.6734021814825324 };
	//u.vals[10] = { 1.703022808958342, 1.316354398929080 };
	//u.vals[11] = { 0.7784162545009661, -1.704706038580704 };
	//u.vals[12] = { 1.330862634637914, -1.550806611706378 };
	//u.vals[13] = { -0.5232538905713433, -0.4784752090431956 };
	//u.vals[14] = { -1.321976965402019, 1.346809599650066 };
	//u.vals[15] = { 0.5990914872549524, -0.6198521954379093 };
	//u.vals[16] = { -1.085768780299999, 1.927095898857544 };
	//u.vals[17] = { 1.477553981971008, -1.791283585389708 };
	//u.vals[18] = { -0.2134766734865376, 0.8082908720352480 };
	//u.vals[19] = { -0.2336714640220814, -0.5763119044327443 };
	//u.vals[20] = { -0.5224397215065819, -1.598939225974838 };
	//u.vals[21] = { 0.6854885190971753, -0.3408027730110907 };
	//u.vals[22] = { 0.7645178222346232, 1.186191041599710 };


	//Vect<nV + 2> u2;
	//u2.vals[0] = { -1.591166140052850, -0.04940904861752138 };
	//u2.vals[1] = { -0.01771927296179410, -1.757536693495840 };
	//u2.vals[2] = { 0.7335430195676036, 1.494677560601192 };
	//u2.vals[3] = { -0.3361337363044399, -1.964058650705113 };
	//u2.vals[4] = { 1.330795013228213, -1.573548731243773 };
	//u2.vals[5] = { 0.7108337719736726, 1.629612214605046 };
	//u2.vals[6] = { 0.3928209446059512, -1.149517855134655 };
	//u2.vals[7] = { -0.6241878141285646, 0.6375838975053414 };
	//u2.vals[8] = { 0.3617447751658736, 1.769385686440584 };
	//u2.vals[9] = { 1.536913610358482, -0.6638364362934164 };
	//u2.vals[10] = { 1.618078831968282, 1.789326249870561 };
	//u2.vals[11] = { -0.7026756055762897, -1.236378280382556 };
	//u2.vals[12] = { -1.595358088678271, -0.9494335840556092 };
	//u2.vals[13] = { -0.1157914997686307, -0.1278708428454376 };
	//u2.vals[14] = { -0.9905121051585128, 1.082710870150010 };
	//u2.vals[15] = { -0.6567000956051130, 0.2591163848714890 };
	//u2.vals[16] = { -1.898205143740682, 0.03710724576242175 };
	//u2.vals[17] = { 0.3334507282921590, 0.5222037955069991 };
	//u2.vals[18] = { 0.1984424017011959, 0.06923563142430122 };
	//u2.vals[19] = { 1.174682457284455, 1.093001682827299 };
	//u2.vals[20] = { -0.8270368079948636, -1.091622991185837 };
	//u2.vals[21] = { 0.1041016121144578, -0.9933154550096672 };
	//u2.vals[22] = { 1.836351343935967, -0.2532385004005082 };


}



int main(int argc, char *argv[]){

	/*----------Enter number of rounds to compute----------*/
	int nRounds;
	if (argc < 1 || sscanf(argv[1], "%i", &nRounds) != 1) {
		std::cout << "nRound input error" << std::endl; return;
	}


	/*----------Display CUDA information----------*/
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	std::cout << "GPU name: " << properties.name << std::endl;
	std::cout << "Compute capability: " << properties.major << "." << properties.minor << std::endl;
	std::cout << "Streaming multiprocessors: " << properties.multiProcessorCount << std::endl;

	std::cout << "CUDA Toolkit version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << std::endl;

	/*----------Define final parameters----------*/
	Vect<nP> *fpm = (Vect<nP>*)malloc(sizeof(Vect<nP>));

	fpm[0].vals[0] = { 4.241713408139979, -4.925334736425448 };
	fpm[0].vals[1] = { 1.448005491494222, -2.686249989236069 };
	fpm[0].vals[2] = { 2.287911194543938, -0.9123108488455074 };
	fpm[0].vals[3] = { -1.870502483790705, -0.5778077414588303 };
	fpm[0].vals[4] = { -0.7625329418400497, 0.9043788195368165 };
	fpm[0].vals[5] = { -4.142822753796651, 1.697671322829592 };
	fpm[0].vals[6] = { 1.023004534182935, -3.026210035710708 };
	fpm[0].vals[7] = { -2.522295755933046, 3.942421532145637 };
	fpm[0].vals[8] = { -3.769870484797034, 0.7972377857916531 };
	fpm[0].vals[9] = { -4.285631161563732, -0.5650246890525263 };
	fpm[0].vals[10] = { -2.166030800874488, 0.8836055774820508 };
	fpm[0].vals[11] = { 4.579260949000442, -2.074013737196861 };
	fpm[0].vals[12] = { 4.801289733755713, -1.460721841824302 };
	fpm[0].vals[13] = { 2.041266673362720, 2.962149232597863 };
	fpm[0].vals[14] = { 3.989999692714399, -0.2328295911150420 };
	fpm[0].vals[15] = { -3.996448356004510, 4.426854856711802 };
	fpm[0].vals[16] = { 2.315936719955769, -0.5527803857724418 };
	fpm[0].vals[17] = { -3.018911857947673, 0.9692384144638879 };
	fpm[0].vals[18] = { -0.4906347357049512, 2.994316903874402 };
	fpm[0].vals[19] = { 0.2397118512793472, -1.341440197750693 };
	fpm[0].vals[20] = { 4.942917423339324, -2.415216683074435 };
	fpm[0].vals[21] = { 1.599536873563615, 2.367319121024760 };
	fpm[0].vals[22] = { 4.229853181495468, -1.149104287738972 };
	fpm[0].vals[23] = { -0.2180298061101595, 3.585004213517870 };
	fpm[0].vals[24] = { 4.311063203842579, -0.03375595047392820 };
	fpm[0].vals[25] = { 4.938148066901551, 4.551701974206413 };
	fpm[0].vals[26] = { -4.271214442569748, 0.1355349863802999 };
	fpm[0].vals[27] = { -0.7377434797048270, 1.831738837919296 };
	fpm[0].vals[28] = { 0.8894614668331116, 0.8657473695398501 };
	fpm[0].vals[29] = { 4.036757863569754, 1.461164684969592 };



	/*----------GPU variables----------*/

	Vect<nV> *fpt = (Vect<nV>*)malloc(sizeof(Vect<nV>) * nPaths); // the final points output during each round of GPU computing
	int *errorCodes = (int*)malloc(sizeof(int)*nPaths); // corresponding errorCodes output during each round of GPU computing

	Vect<nV + 2> *vectDebug = (Vect<nV+2>*)malloc(sizeof(Vect<nV + 2>) * nPaths); // for debugging
	Matr<nV + 2, nV + 2> *matrDebug = (Matr<nV + 2, nV + 2>*)malloc(sizeof(Matr<nV + 2, nV + 2>)*nPaths);

	// Some variables for timing
	float timingTracking;
	cudaEvent_t timingCudaStart, timingCudaStop;
	cudaEventCreate(&timingCudaStart);
	cudaEventCreate(&timingCudaStop);


	// Start timing
	cudaEventRecord(timingCudaStart, 0);


	//Allocate device memory

	cudaError_t stat = cudaSuccess;

	curandState *d_randStates;
	stat = cudaMalloc(&d_randStates, sizeof(curandState)*nPaths);
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);

	Vect<nP> *d_fpm;
	stat = cudaMalloc(&d_fpm, sizeof(Vect<nP>));
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);
	stat = cudaMemcpy(d_fpm, fpm, sizeof(Vect<nP>), cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);

	Vect<nV> *d_fpt;
	stat = cudaMalloc(&d_fpt, sizeof(Vect<nV>)*nPaths);
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);

	int *d_errorCodes;
	stat = cudaMalloc(&d_errorCodes, sizeof(int)*nPaths);
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);

	Vect<nV + 2> *d_vectDebug;
	stat = cudaMalloc(&d_vectDebug, sizeof(Vect<nV + 2>)*nPaths);
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);

	Matr<nV + 2, nV + 2> *d_matrDebug;
	stat = cudaMalloc(&d_matrDebug, sizeof(Matr<nV + 2, nV + 2>)*nPaths);
	std::cout << cudaGetErrorName(stat) << std::endl;
	assert(stat == cudaSuccess);


	// End timing
	cudaEventRecord(timingCudaStop, 0);
	cudaEventSynchronize(timingCudaStop);
	cudaEventElapsedTime(&timingTracking, timingCudaStart, timingCudaStop);
	std::cout << timingTracking * 0.001 << " sec to allocate memory" << std::endl;





	/*----------Analysis variables-----------*/

	thrust::device_vector<Real> d_firstElems(nPaths); // device space for the first elements of fpt
	thrust::device_vector<int> d_NtoTnew(nPaths); // device space for the ordering of fpt
	int *NtoTnew = (int*)malloc(sizeof(int)*nPaths); // host space for the ordering of fpt

	std::deque<Vect<nV>> fptColl; // sorted roots collected over GPU rounds, size == nRounds*nPaths - duplicates - errors
	std::vector<int> errorCodesAgr; // errorCodes from every trial, size == nRounds*nPaths

	// Progress statistics
	int successesRound = 0;
	int duplicatesRound = 0;
	int trackingErrorsRound = 0;
	int err1Round = 0;
	int err2Round = 0;
	int err3Round = 0;
	int err4Round = 0;
	int err5Round = 0;

	int successesTotal = 0;
	int duplicatesTotal = 0;
	int trackingErrorsTotal = 0;
	int err1Total = 0;
	int err2Total = 0;
	int err3Total = 0;
	int err4Total = 0;
	int err5Total = 0;

	int round = 0;

	Real successRatio = 1.;
	Real PercRootsEst = 0.;

	// Timing variables
	std::chrono::steady_clock::time_point timingStart;
	std::chrono::steady_clock::time_point timingEnd;
	double timingDelDup;
	double timingFileWrites;

	// File streams
	std::ofstream fptCollFile;
	std::ofstream errorCodesAgrFile;
	std::ofstream progressFile;

	/*----------Check for pre-existing files to load----------*/

	// fptColl
	std::ifstream fptCollFileLoad("fptColl", std::ios::in | std::ios::binary | std::ios::ate);
	if (fptCollFileLoad.is_open()){ // check if file is present

		// Load previously collected roots
		int nfptColl = fptCollFileLoad.tellg() / sizeof(Vect<nV>); // number of previously collected roots
		fptCollFileLoad.seekg(0, std::ios::beg); // set get position to the beginning
		fptColl.resize(nfptColl); // allocate space
		for (std::deque<Vect<nV>>::iterator root = fptColl.begin(); root != fptColl.end(); ++root){
			fptCollFileLoad.read((char*)&(*root), sizeof(Vect<nV>)); // read in input data
		}
		fptCollFileLoad.close(); // close file
		std::cout << "Previous roots collected: " << nfptColl << std::endl;
	}

	// errorCodesAgr
	std::ifstream errorCodesAgrFileLoad("errorCodesAgr", std::ios::in | std::ios::binary | std::ios::ate);
	if (errorCodesAgrFileLoad.is_open()){ // check if file is present

		// Load previous error codes
		int trialsTotal = errorCodesAgrFileLoad.tellg() / sizeof(int); // number of trials found in the previous solution file
		errorCodesAgrFileLoad.seekg(0, std::ios::beg); // set get position to the beginning
		errorCodesAgr.reserve(trialsTotal); // allocate...
		errorCodesAgr.resize(trialsTotal); // ...space
		errorCodesAgrFileLoad.read((char*)&errorCodesAgr[0], sizeof(int)*trialsTotal); // read in input data
		errorCodesAgrFileLoad.close(); // close file
		std::cout << "Previous trials: " << trialsTotal << std::endl;

		// Count previous error codes
		for (std::vector<int>::iterator code = errorCodesAgr.begin(); code != errorCodesAgr.end(); ++code){
			if (*code == 0){ successesTotal++; }
			else if (*code == 1){ err1Total++; }
			else if (*code == 2){ err2Total++; }
			else if (*code == 3){ err3Total++; }
			else if (*code == 4){ err4Total++; }
			else if (*code == 5){ err5Total++; }
			else if (*code == 6){ duplicatesTotal++; }
		}
		trackingErrorsTotal = err1Total + err2Total + err3Total + err4Total + err5Total;
		round = trialsTotal / nPaths;
		std::cout << "Previous rounds: " << round << std::endl;
	}









	std::cout << std::endl;


	/*----------Loop----------*/

	for (int r=1; r <= nRounds; ++r){

		std::cout << "Round " << round + r << std::endl;

		// Start timing
		cudaEventRecord(timingCudaStart, 0);

		/*----------Initialize random number generators----------*/
		unsigned long long seed = (unsigned long long)time(NULL);
		//unsigned long long seed = 1300; 
		InitRandStates<<<nBlocks, nThreads>>>(d_randStates, seed);
		cudaDeviceSynchronize();
		stat = cudaGetLastError();
		if (stat != cudaSuccess){
			std::cout << "Random number generator failure" << std::endl;
			std::cout << "kernel error: " << stat << std::endl;
			std::cout << "kernel error name: " << cudaGetErrorName(stat) << std::endl;
			std::cout << "kernel error description: " << cudaGetErrorString(stat) << std::endl << std::endl;
			return;
		}


		/*----------Run the main kernel----------*/
		Homotopy<<<nBlocks, nThreads>>>(d_randStates, d_fpm, d_fpt, d_errorCodes, d_vectDebug, d_matrDebug);
		cudaDeviceSynchronize();
		stat = cudaGetLastError();
		if (stat == cudaSuccess){ std::cout << "Homotopy GPU success" << std::endl; }
		else{
			std::cout << "!!! Homotopy GPU failure !!!" << std::endl;
			std::cout << "kernel error: " << stat << std::endl;
			std::cout << "kernel error name: " << cudaGetErrorName(stat) << std::endl;
			std::cout << "kernel error description: " << cudaGetErrorString(stat) << std::endl << std::endl;
			return;
		}


		/*----------Copy back memory----------*/

		stat = cudaMemcpy(fpt, d_fpt, sizeof(Vect<nV>)*nPaths, cudaMemcpyDeviceToHost);
		if (stat != cudaSuccess){
			std::cout << "Error copying fpt back" << std::endl;
			std::cout << "copy back error: " << stat << std::endl;
			std::cout << "copy error name: " << cudaGetErrorName(stat) << std::endl;
			std::cout << "copy error description: " << cudaGetErrorString(stat) << std::endl;
			return;
		}

		stat = cudaMemcpy(errorCodes, d_errorCodes, sizeof(int)*nPaths, cudaMemcpyDeviceToHost);
		if (stat != cudaSuccess){
			std::cout << "Error copying errorCodes back" << std::endl;
			std::cout << "copy back error: " << stat << std::endl;
			std::cout << "copy error name: " << cudaGetErrorName(stat) << std::endl;
			std::cout << "copy error description: " << cudaGetErrorString(stat) << std::endl;
		}

		//stat = cudaMemcpy(vectDebug, d_vectDebug, sizeof(Vect<nV + 2>)*nPaths, cudaMemcpyDeviceToHost);
		//if (stat != cudaSuccess){ printf("!!!!!!!!!!!!!!!!!!!!"); }
		//std::cout << "copy back error: " << stat << std::endl;
		//std::cout << "copy error name: " << cudaGetErrorName(stat) << std::endl;
		//std::cout << "copy error description: " << cudaGetErrorString(stat) << std::endl;
		//assert(stat == cudaSuccess);

		//stat = cudaMemcpy(matrDebug, d_matrDebug, sizeof(Matr<nV + 2, nV + 2>)*nPaths, cudaMemcpyDeviceToHost);
		//if (stat != cudaSuccess){ printf("!!!!!!!!!!!!!!!!!!!!"); }
		//std::cout << "copy back error: " << stat << std::endl;
		//std::cout << "copy error name: " << cudaGetErrorName(stat) << std::endl;
		//std::cout << "copy error description: " << cudaGetErrorString(stat) << std::endl;
		//assert(stat == cudaSuccess);


		// End timing
		cudaEventRecord(timingCudaStop, 0);
		cudaEventSynchronize(timingCudaStop);
		cudaEventElapsedTime(&timingTracking, timingCudaStart, timingCudaStop);
		timingTracking = timingTracking*0.001;
		std::cout << nPaths << " paths tracked in " << timingTracking << " sec" << std::endl << std::endl;


		/*----------Analysis----------*/

		/*Compute ordering of new roots (fpt)*/
		for (int i = 0; i < nPaths; ++i){ d_firstElems[i] = d_fpt[i].vals[0].x; } // copy values into d_firstElems
		thrust::sequence(d_NtoTnew.begin(), d_NtoTnew.end(), 0); // sequence d_NtoTnew as {0, 1, ... , nPaths}
		thrust::sort_by_key(d_firstElems.begin(), d_firstElems.end(), d_NtoTnew.begin()); // sort d_NtoTnew according to the ordering of the first elements
		for (int i = 0; i < nPaths; ++i){ NtoTnew[i] = d_NtoTnew[i]; } // transfer sorted indices to host

		timingStart = std::chrono::steady_clock::now();

		/*Sort, delete duplicates, and merge the new roots (fpt) into the so far collected roots (fptColl)*/
		SortingDeletingMerging(&fptColl, fpt, errorCodes, NtoTnew, nPaths);
		errorCodesAgr.insert(errorCodesAgr.end(), errorCodes, errorCodes + nPaths);

		// Compute progress statistics
		successesRound = duplicatesRound = trackingErrorsRound = err1Round = err2Round = err3Round = err4Round = err5Round = 0;
		for (int i = 0; i < nPaths; ++i){
			if (errorCodes[i] == 0){ successesRound++; }
			else if (errorCodes[i] == 1){ err1Round++; }
			else if (errorCodes[i] == 2){ err2Round++; }
			else if (errorCodes[i] == 3){ err3Round++; }
			else if (errorCodes[i] == 4){ err4Round++; }
			else if (errorCodes[i] == 5){ err5Round++; }
			else if (errorCodes[i] == 6){ duplicatesRound++; }
		}
		successesTotal += successesRound;
		err1Total += err1Round;
		err2Total += err2Round;
		err3Total += err3Round;
		err4Total += err4Round;
		err5Total += err5Round;
		trackingErrorsRound = err1Round + err2Round + err3Round + err4Round + err5Round;
		duplicatesTotal += duplicatesRound;
		trackingErrorsTotal += trackingErrorsRound;
		successRatio = (Real)successesTotal / ((Real)successesTotal + (Real)duplicatesTotal);
		PercRootsEst = successRatio*LambertW((-1 / successRatio)*exp(-1 / successRatio)) + 1;

		timingEnd = std::chrono::steady_clock::now();
		timingDelDup = (double)std::chrono::duration_cast<std::chrono::microseconds>(timingEnd - timingStart).count() * 0.000001;

		timingStart = std::chrono::steady_clock::now();

		/*Write the root collection and aggregate of error codes to files*/
		fptCollFile.open("fptColl", std::ofstream::out | std::ofstream::binary);
		for (std::deque<Vect<nV>>::iterator root = fptColl.begin(); root != fptColl.end(); ++root){
			fptCollFile.write((char*)&(*root), sizeof(Vect<nV>));
		}
		fptCollFile.close();

		errorCodesAgrFile.open("errorCodesAgr", std::ofstream::out | std::ofstream::binary | std::ofstream::app);
		errorCodesAgrFile.write((char*)errorCodes, sizeof(int) * nPaths);
		errorCodesAgrFile.close();

		timingEnd = std::chrono::steady_clock::now();
		timingFileWrites = (double)std::chrono::duration_cast<std::chrono::microseconds>(timingEnd - timingStart).count() * 0.000001;

		/*Write to the progress file*/
		progressFile.open("progress", std::ofstream::out | std::ofstream::app);
		progressFile << "-----Progress statistics for round " << round + r << "-----" << std::endl;
		progressFile << std::endl;
		progressFile << "---In Total---" << std::endl;
		progressFile << "Roots collected:" << std::right << std::setw(10) << successesTotal << std::endl;
		progressFile << "Duplicates:     " << std::right << std::setw(10) << duplicatesTotal << std::endl;
		progressFile << "Trials:         " << std::right << std::setw(10) << errorCodesAgr.size() << std::endl;
		progressFile << "Tracking errors:" << std::right << std::setw(10) << trackingErrorsTotal;
		progressFile << "    (" << err1Total << " " << err2Total << " " << err3Total << " " << err4Total << " " << err5Total << ")" << std::endl;
		progressFile << std::endl;
		progressFile << "---Per Round---" << std::endl;
		progressFile << "Roots collected:" << std::right << std::setw(10) << successesRound << std::endl;
		progressFile << "Duplicates:     " << std::right << std::setw(10) << duplicatesRound << std::endl;
		progressFile << "Trials:         " << std::right << std::setw(10) << nPaths << std::endl;
		progressFile << "Tracking errors:" << std::right << std::setw(10) << trackingErrorsRound;
		progressFile << "    (" << err1Round << " " << err2Round << " " << err3Round << " " << err4Round << " " << err5Round << ")" << std::endl;
		progressFile << std::endl;
		progressFile << "---Progress---" << std::endl;
		progressFile << "Success ratio:          " << std::right << std::setw(10) << successRatio << std::endl;
		progressFile << "Roots collected (est.): " << std::right << std::setw(10) << PercRootsEst * 100 << "%" << std::endl;
		progressFile << std::endl;
		progressFile << "--- Timing ---" << std::endl;
		progressFile << "Tracking:                   " << std::right << std::setw(10) << timingTracking << " sec" << std::endl;
		progressFile << "Sorting, del. dup., merging:" << std::right << std::setw(10) << timingDelDup << " sec" << std::endl;
		progressFile << "Writing to disk:            " << std::right << std::setw(10) << timingFileWrites << " sec" << std::endl;
		progressFile << std::endl << std::endl;
		progressFile.close();

	}



	cudaFree(d_randStates);
	cudaFree(d_fpm);
	cudaFree(d_fpt);
	cudaFree(d_errorCodes);
	cudaFree(d_vectDebug);
	cudaFree(d_matrDebug);


	free(fpm);
	free(fpt);
	free(errorCodes);
	free(NtoTnew);
	free(vectDebug);
	free(matrDebug);






	//std::list<Vect<nV>> fptAgrlist;

	//// Copy fptAgr (vector) into a list
	//int chunk = 1000; // chunks are copied in, then that chunk is deleted
	//int iter_start;
	//if (nfptAgr % chunk == 0){ iter_start = nfptAgr - chunk; } 
	//else{ iter_start = nfptAgr - nfptAgr % chunk; } // iterating over {0, 1, ... , 13} with chunk size of 3 will proceed i = 12, 9, 6, 3, 0
	//for (int i = iter_start; i >= 0; i -= chunk){
	//	fptAgrlist.insert(fptAgrlist.begin(), fptAgr.begin() + i, fptAgr.end()); // insert chunk at the beginning of the list
	//	fptAgr.erase(fptAgr.begin() + i, fptAgr.end()); // erase chunk from the original vector
	//	fptAgr.shrink_to_fit(); // shrink the vector's capacity
	//}



}
