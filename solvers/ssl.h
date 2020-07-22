/*    Copyright 2006 Vikas Sindhwani (vikass@cs.uchicago.edu)
			SVM-lin: Fast SVM Solvers for Supervised and Semi-supervised Learning

			This file is part of SVM-lin.      

			SVM-lin is free software; you can redistribute it and/or modify
			it under the terms of the GNU General Public License as published by
			the Free Software Foundation; either version 2 of the License, or
			(at your option) any later version.
 
			SVM-lin is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
			MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
			GNU General Public License for more details.

			You should have received a copy of the GNU General Public License
			along with SVM-lin (see gpl.txt); if not, write to the Free Software
			Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/
#ifndef _svmlin_H
#define _svmlin_H
#include <vector>
#include <ctime>

using namespace std;

/* OPTIMIZATION CONSTANTS */
#define CGITERMAX 10000 /* maximum number of CGLS iterations */
#define SMALL_CGITERMAX 10 /* for heuristic 1 in reference [2] */
#define EPSILON   1e-7 /* most tolerances are set to this value */
#define BIG_EPSILON 0.01 /* for heuristic 2 in reference [2] */
#define RELATIVE_STOP_EPS 1e-9 /* for L2-SVM-MFN relative stopping criterion */
#define MFNITERMAX 50 /* maximum number of MFN iterations */

#define VERBOSE_CGLS 0
        extern "C" struct data 
	{
	  int m; /* number of examples */
	  int n; /* number of features */ 
	  double* X; // flattened dense feature matrix
	  double *Y;   /* labels */
	  double cpos;   /* cost associated with each positive example */
	  double cneg;   /* cost associated with each negative example */
	};

	extern "C" struct vector_double /* defines a vector of doubles */
	{
		int d; /* number of elements */
		double *vec; /* ptr to vector elements*/
	};



	extern "C" struct vector_int /* defines a vector of ints for index subsets */
	{
		int d; /* number of elements */
		int *vec; /* ptr to vector elements */
	};

class timer { /* to output run time */
protected:
	double start, finish;
public:
	vector<double> times;
	void record() {
		times.push_back(time());
	}
	void reset_vectors() {
		times.erase(times.begin(), times.end());
	}
	void restart() { start = clock(); }
	void stop() { finish = clock(); }
	double time() const { return ((double)(finish - start))/CLOCKS_PER_SEC; }
};

class Delta { /* used in line search */
 public: 
	 Delta() {delta=0.0; index=0;s=0;};  
	 double delta;   
	 int index;
	 int s;   
};
inline bool operator<(const Delta& a , const Delta& b) { return (a.delta < b.delta);};

extern "C" void clear_data(struct data *a); /* deletes a */
extern "C" void call_L2_SVM_MFN(double* X, double* Y, double* w,
				     double cpos, double cneg,
				     int n, // num features
				     int m, // num instances
				     double labmda = 1, 
				     int verbose = 0);

/* svmlin algorithms and their subroutines */
 
/* Conjugate Gradient for Sparse Linear Least Squares Problems */
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_{i in Subset} Data->C[i] (Y[i]- w' x_i)^2 */
/* over a subset of examples x_i specified by vector_int Subset */
int CGLS(const struct data *Data, 
	 const int cgitermax,
         const double epsilon,
	 const struct vector_int *Subset, 
	 double* beta,
	 double* o,
	 int verbose, double cpos, double cneg);

/* Linear Modified Finite Newton L2-SVM*/
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_i Data->C[i] max(0,1 - Y[i] w' x_i)^2 */
int L2_SVM_MFN(const struct data *Data, 
	       double* w, double* o,
	       int verbose, 
	       double cpos, double cneg, 
	       double lambda_l); /* use ini=0 if no good starting guess for Weights, else 1 */
double line_search(double *w, 
                   double *w_bar,
                   double lambda_l,
                   double *o, 
                   double *o_bar, 
                   double *Y, 
                   int d, /* data dimensionality -- 'n' */
                   int l,
		   double cpos, double cneg);


#endif
