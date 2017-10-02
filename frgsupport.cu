#include <vector>
#include <list>
#include <deque>

#include "htpykinematics.cuh"

#define Tol 0.000001 // tolerance to test whether elements are the same
#define dupCode 6 // error code associated with a duplicate

void SortingDeletingMerging(std::deque<Vect<nV>> *fptColl, Vect<nV> *fpt, int *errorCodes, int *NtoTnew, int nPaths){

	// This function merge-sorts the new roots (fpt) into the current collection (fptColl), 
	// simultaneously deleting duplicates and all other errors.  As well, errorCodes are updated marking 
	// duplicates with a 6.  Nothing is deleted from errorCodes so that its size remains nPaths.

	// fptColl: (in/out) the current sorted collection of roots
	// fpt: (in) the new roots furnished by the GPU computation
	// errorCodes: (in/out) elements correspond to fpt, 0: no error, 1-5: path tracking errors, 6: duplicates (marked in this function)
	// NtoTnew: (in) the ordering of fpt by the real component of its first element
	// nPaths: (in) the size of fpt

	int nfptColl = (*fptColl).size(); // number of collected roots

	std::list<int> NtoTlist; // Populate NtoTlist with the sequential ordered indices of fptColl i.e. {0, 1, 2, ... , nfptColl}
	for (int i = 0; i < nfptColl; ++i){ NtoTlist.push_back(i); }


	// Merge NtoTnew into NtoTlist
	// this modifies NtoTlist to have size of nfptColl+nPaths
	// NtoTlist orig: {0, 1, 2, 3}
	// NtoTnew: {5, 6, 4}
	// NtoTlist mod: {0, 5, 6, 1, 2, 4, 3}
	// inserts:          ^  ^        ^
	int inew = 0;
	for (std::list<int>::iterator i = NtoTlist.begin(); i != NtoTlist.end(); ++i){ // iterate through NtoTlist
		
		while (fpt[NtoTnew[inew]].vals[0].x < (*fptColl)[*i].vals[0].x && inew < nPaths){ // insert between elements of NtoTlist
			NtoTlist.insert(i, NtoTnew[inew] + nfptColl);
			inew++;
		}
	}
	if (inew < nPaths){ // insert after elements of NtoTlist
		for (int i = inew; i < nPaths; ++i){ NtoTlist.push_back(NtoTnew[i] + nfptColl); }
	}


	// Since NtoTlist is an std::list, we do not have efficient access to its elements
	// Transfer NtoTlist into vector NtoT, clear NtoTlist
	std::vector<int> NtoT;
	NtoT.reserve(NtoTlist.size());
	NtoT.resize(NtoTlist.size());
	std::copy(make_move_iterator(NtoTlist.begin()), make_move_iterator(NtoTlist.end()), NtoT.begin());
	NtoTlist.clear();


	// Construct TtoN from NtoT
	std::vector<int> TtoN;
	TtoN.reserve(NtoT.size());
	TtoN.resize(NtoT.size());
	for (int i = 0; i < NtoT.size(); ++i){ TtoN[NtoT[i]] = i; }

	// trial = NtoT[num_rank] ,  input: numerical rank,  output: trial no.
	// num_rank = TtoN[trial] ,  input: trial no.,  output: numerical rank




	/*----------Begin Mark Duplicates in errorCodes----------*/
	for (int tloc = 0; tloc < nPaths; ++tloc){ // iterate through local trial numbers contained in fpt

		if (errorCodes[tloc] == 0){ // only check for duplicates on roots that did not throw an error

			int tglo = tloc + nfptColl; // obtain a global trial number
			int n = TtoN[tglo]; // global rank of t



			/*---Begin Left Hand Neighbors---*/
			for (int i = n - 1; i >= 0; --i){ // iterate from left hand neighbor to the beginning

				int uglo = NtoT[i]; // obtain global trial number of neighbor
				int uloc = uglo - nfptColl; // obtain local trial number of neighbor

				/*---If first elements don't match, we are out of the neighborhood---*/
				if (uloc < 0){ // if the neighbors index is in the collected roots
					if (abs(fpt[tloc].vals[0].x - (*fptColl)[uglo].vals[0].x) > Tol){ break; }
				}
				else{ // if the neighbors index is in the new roots
					if (abs(fpt[tloc].vals[0].x - fpt[uloc].vals[0].x) > Tol){ break; }
				}

				/*---Otherwise, check and mark duplications---*/
				if (uloc < 0){ // if the neighbors index is in the collected roots

					// check if t and the neighbor are the same
					bool sameQ = true;
					for (int j = 0; j < nV; ++j){
						if (abs(fpt[tloc].vals[j].x - (*fptColl)[uglo].vals[j].x) > Tol){ sameQ = false; break; }
						if (abs(fpt[tloc].vals[j].y - (*fptColl)[uglo].vals[j].y) > Tol){ sameQ = false; break; }
					}
					if (sameQ){ errorCodes[tloc] = dupCode; } // if they are the same, mark trial as a duplicate

				}
				else{ // if neighbor has index in the new roots

					if (uloc < tloc && errorCodes[uloc] == 0){ // check if the neighbor has occurred && is not already marked as a duplicate of otherwise invalid

						// check if t and the neighbor are the same
						bool sameQ = true;
						for (int j = 0; j < nV; ++j){
							if (abs(fpt[tloc].vals[j].x - fpt[uloc].vals[j].x) > Tol){ sameQ = false; break; }
							if (abs(fpt[tloc].vals[j].y - fpt[uloc].vals[j].y) > Tol){ sameQ = false; break; }
						}
						if (sameQ){ errorCodes[tloc] = dupCode; } // if they are the same, mark trial as a duplicate

					}
				}

			} /*---End Left Hand Neighbors---*/





			/*---Begin Right Hand Neighbors---*/
			for (int i = n + 1; i < nfptColl + nPaths; ++i){ // iterate from right hand neighbor to the end

				int uglo = NtoT[i]; // obtain global trial number of neighbor
				int uloc = uglo - nfptColl; // obtain local trial number of neighbor

				/*---If first elements don't match, we are out of the neighborhood---*/
				if (uloc < 0){ // if the neighbors trial number is in the collected roots
					if (abs(fpt[tloc].vals[0].x - (*fptColl)[uglo].vals[0].x) > Tol){ break; }
				}
				else{ // if the neighbors trial number is in the new roots
					if (abs(fpt[tloc].vals[0].x - fpt[uloc].vals[0].x) > Tol){ break; }
				}

				/*---Otherwise, check and mark duplications---*/
				if (uloc < 0){ // if the neighbors index is in the collected roots

					// check if t and the neighbor are the same
					bool sameQ = true;
					for (int j = 0; j < nV; ++j){
						if (abs(fpt[tloc].vals[j].x - (*fptColl)[uglo].vals[j].x) > Tol){ sameQ = false; break; }
						if (abs(fpt[tloc].vals[j].y - (*fptColl)[uglo].vals[j].y) > Tol){ sameQ = false; break; }
					}
					if (sameQ){ errorCodes[tloc] = dupCode; } // if they are the same, mark trial as a duplicate

				}
				else{ // if neighbor has index in the new roots

					if (uloc < tloc && errorCodes[uloc] == 0){ // check if the neighbor has occurred && is not already marked as a duplicate of otherwise invalid

						// check if t and the neighbor are the same
						bool sameQ = true;
						for (int j = 0; j < nV; ++j){
							if (abs(fpt[tloc].vals[j].x - fpt[uloc].vals[j].x) > Tol){ sameQ = false; break; }
							if (abs(fpt[tloc].vals[j].y - fpt[uloc].vals[j].y) > Tol){ sameQ = false; break; }
						}
						if (sameQ){ errorCodes[tloc] = dupCode; } // if they are the same, mark trial as a duplicate

					}
				}

			} /*---End Right Hand Neighbors---*/



		}
	}
	/*----------End Mark Duplicates in errorCodes----------*/



	// Merge new roots (fpt) into collected roots (fptColl)
	// Merged fptColl is sorted and contains no duplicates or any root that threw an error code
	// This is accomplished efficiently by "folding" into the deque
	/*Example*/
	// fptColl orig:   {r0, r1, r2, r3}
	// fpt:            {r5, r6, r4}
	// fptColl target: {r0, r5, r6, r1, r2, r4, r3}
	// move front to back:    {r1, r2, r3, r0}
	// insert new element:    {r1, r2, r3, r0}  <-- r5
	// insert new element:    {r1, r2, r3, r0, r5} <-- r6
	// new elements inserted: {r1, r2, r3, r0, r5, r6}
	// move front to back:    {r3, r0, r5, r6, r1, r2}
	// insert new element:    {r3, r0, r5, r6, r1, r2} <-- r4
	// new element inserted:  {r3, r0, r5, r6, r1, r2, r4}
	// move front to back:    {r0, r5, r6, r1, r2, r4, r3}
	{int i = 0;
	while (i < NtoT.size()){ // forward iterate through the ordering

		int range = 0; // measure the length of an ordering sequence below nfptColl (aka belongs to fptColl)
		while (NtoT[i] + 1 == NtoT[i + 1] && NtoT[i + 1] < nfptColl){ range++; i++; }
		// example: a sequence of 3 elements will measure range=2

		if (range > 0){ // if a sequence greater than 1 is detected
			(*fptColl).insert((*fptColl).end(), (*fptColl).begin(), (*fptColl).begin() + range); // move it from the front of the deque to the back
			(*fptColl).erase((*fptColl).begin(), (*fptColl).begin() + range); // delete is from the front
		}
		else if (NtoT[i] < nfptColl){ // if the sequence is only 1 element, but is below nfptColl (belongs to fptColl)
			(*fptColl).push_back((*fptColl).front()); // move it from the front to the back
			(*fptColl).pop_front(); // delete it from the front
		}
		else if (errorCodes[NtoT[i] - nfptColl] == 0){ // if the element belongs to the new roots (fpt) and did not throw an error code
			(*fptColl).push_back(fpt[NtoT[i] - nfptColl]); // append it to the end fptColl
		}

		if (range == 0){ i++; }
	}}



}




// Lambert function from https://sites.google.com/site/istvanmezo81/others

//z * exp(z)
Real zexpz(Real z){ return z * exp(z); }

//The derivative of z * exp(z) = exp(z) + z * exp(z)
Real zexpz_d(Real z){ return exp(z) + z * exp(z); }

//The second derivative of z * exp(z) = 2. * exp(z) + z * exp(z)
Real zexpz_dd(Real z){ return 2. * exp(z) + z * exp(z); }


//Determine the initial point for the root finding
Real InitPoint(Real z){

	const Real e{ 2.71828182845904523536 };
	Real ip{ log(z) - log(log(z)) };// initial point coming from the general asymptotic approximation
	Real p{ sqrt(2. * (e * z + 1.)) };// used when we are close to the branch cut around zero and when k=0,-1

	if (abs(z - (-exp(-1.))) <= 1.){ //we are close to the branch cut, the initial point must be chosen carefully
		ip = -1. + p - 1. / 3. * pow(p, 2) + 11. / 72. * pow(p, 3);
	}

	if (abs(z - .5) <= .5) ip = (0.35173371 * (0.1237166 + 7.061302897 * z)) / (2. + 0.827184 * (1. + 2. * z));// (1,1) Pade approximant for W(0,a)

	return ip;
}


Real LambertW(Real z){

	int k = 0;

	//For some particular z and k W(z,k) has simple value:
	if (z == 0.) return (k == 0) ? 0. : -INFINITY;
	if (z == -exp(-1.) && (k == 0 || k == -1)) return -1.;
	if (z == exp(1.) && k == 0) return 1.;

	//Halley method begins
	Real w{ InitPoint(z) }, wprev{ InitPoint(z) }; // intermediate values in the Halley method
	const unsigned int maxiter = 30; // max number of iterations. This eliminates improbable infinite loops
	unsigned int iter = 0; // iteration counter
	Real prec = 1.E-30; // difference threshold between the last two iteration results (or the iter number of iterations is taken)

	do
	{
		wprev = w;
		w -= 2.*((zexpz(w) - z) * zexpz_d(w)) /
			(2.*pow(zexpz_d(w), 2) - (zexpz(w) - z)*zexpz_dd(w));
		iter++;
	} while ((abs(w - wprev) > prec) && iter < maxiter);
	return w;
}

