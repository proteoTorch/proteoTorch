/*    Copyright 2006 Vikas Sindhwani (vikass@cs.uchicago.edu)
	  Modified  2014 Phong Vo        (phong.vodinh@gmail.com, dinphong.vo@cea.fr)
	  
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

	/* Data: Input examples are stored in sparse (Compressed Row Storage) format */
	extern "C" struct data 
	{
	  int m; /* number of examples */
	  int l; /* number of labeled examples */
	  int u; /* number of unlabeled examples l+u = m */
	  int n; /* number of features */ 
	  double* X; // flattened dense feature matrix
	  double *Y;   /* labels */
	  double *C;   /* cost associated with each example */
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

	extern "C" struct options 
	{
		/* user options */
		int algo; /* 1 to 4 for RLS,SVM,TSVM,DASVM */
		double lambda_l; /* regularization parameter */
		double lambda_u; /* regularization parameter over unlabeled examples */
		int S; /* maximum number of TSVM switches per fixed-weight label optimization */
		double R; /* expected fraction of unlabeled examples in positive class */
		double Cp; /* cost for positive examples */
		double Cn; /* cost for negative examples */
		/*  internal optimization options */    
		double epsilon; /* all tolerances */
		int cgitermax;  /* max iterations for CGLS */
		int mfnitermax; /* max iterations for L2_SVM_MFN */
		
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

//extern "C" void init_data(struct data *Data, )
extern "C" void init_vec_double(struct vector_double *A, int k, double a);  
	/* initializes a vector_double to be of length k, all elements set to a */
extern "C" void init_vec_int(struct vector_int *A, int k); 
	/* initializes a vector_int to be of length k, elements set to 1,2..k. */
extern "C" void clear_data(struct data *a); /* deletes a */
extern "C" void clear_vec_double(struct vector_double *a); /* deletes a */
extern "C" void clear_vec_int(struct vector_int *a); /* deletes a */

extern "C" 	void ssl_train(struct data *Data, 
					 struct options *Options,
					 struct vector_double *W, /* weight vector */
					 struct vector_double *O,
					 int verbose); /* output vector */

/* svmlin algorithms and their subroutines */
 
/* Conjugate Gradient for Sparse Linear Least Squares Problems */
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_{i in Subset} Data->C[i] (Y[i]- w' x_i)^2 */
/* over a subset of examples x_i specified by vector_int Subset */
int CGLS(const struct data *Data, 
	 const int cgitermax,
         const double epsilon,
	 const struct vector_int *Subset,
	 struct vector_double *Weights,
	 struct vector_double *Outputs,
	 int verbose);

/* Linear Modified Finite Newton L2-SVM*/
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_i Data->C[i] max(0,1 - Y[i] w' x_i)^2 */
int L2_SVM_MFN(const struct data *Data, 
	       struct options *Options, 
	       struct vector_double *Weights,
	       struct vector_double *Outputs,
	       int verbose); /* use ini=0 if no good starting guess for Weights, else 1 */
double line_search(double *w, 
									 double *w_bar,
									 double lambda,
									 double *o, 
									 double *o_bar, 
									 double *Y, 
									 double *C,
									 int d,
									 int l);


#endif
