#include "waxpby.h"
#include <immintrin.h>
#include <omp.h>

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


int waxpby (const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w) {  
  
  int i;
  int loopFactor = 2;
  int loopN = (n/loopFactor)*loopFactor;
  
  __m128d vectorB = _mm_set1_pd(beta);
  __m128d vectorA = _mm_set1_pd(alpha);

  #pragma omp for
  for(i=0; i < loopN; i += loopFactor){
    __m128d vectorX = _mm_load_pd(x+i);
    __m128d vectorY = _mm_load_pd(y+i);
    __m128d vectorXa = _mm_mul_pd(vectorX, vectorA);
    __m128d vectorYb = _mm_mul_pd(vectorY, vectorB);
    __m128d vectorW = _mm_add_pd(vectorYb, vectorXa);
    _mm_store_pd(w+i, vectorW);
  }
  return 0;
}
//The loop is very inefficient because the else encapsulates the whole loop essentially