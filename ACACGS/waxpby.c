#include "waxpby.h"
#include "immintrin.h"

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */



//leave n alone
//need to do loop unrolling, means creating a loop factor and incrementing i by the loop factor each iteration
//double is 64 bits, so can only fit two into a 128 bit vector, use this fact to find a loop factor

int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {  
  //creating loopN and loopFactor
  int loopFactor = 2;
  int loopN = (n/loopFactor)*loopFactor;

  if (alpha==1.0) {
    int i;
    for (int i=0; i<loopN; i+=loopFactor) {
      w[i] = x[i] + beta * y[i];
      w[i+1] = x[i+1] + beta * y[i+1];
    }
    //remaining elements
    for (; i<n; i++){
      w[i] = x[i] + beta + y[i];
    }

  } else if(beta==1.0) {
    int i;
    for (int i=0; i<loopN; i+=loopFactor) {
      w[i] = alpha * x[i] + y[i];
      w[i+1] = alpha * x[i+1] + y[i+1];
    }
    for (; i<n; i++) {
      w[i] = alpha * x[i] + y[i];
    }

  } else {
    int i;
    for (int i=0; i<loopN; i+=loopFactor) {
      w[i] = alpha * x[i] + beta * y[i];
      w[i+1] = alpha * x[i] + beta * y[i+1];
    }
    for (; i<n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }

  return 0;
}
