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


int waxpby (const int n, const float alpha, const float * const x, const float beta, const float * const y, float * const w) {  
  
  int i;
  int loopFactor = 8;
  int loopN = (n/loopFactor)*loopFactor;
  
  __m256 vectorB = _mm256_set1_ps(beta);
  __m256 vectorA = _mm256_set1_ps(alpha);

  #pragma omp parallel for
  for(i=0; i < loopN; i += loopFactor){
    __m256 vectorX = _mm256_load_ps(x+i);
    __m256 vectorY = _mm256_load_ps(y+i);
    __m256 vectorXa = _mm256_mul_ps(vectorX, vectorA);
    __m256 vectorYb = _mm256_mul_ps(vectorY, vectorB);
    __m256 vectorW = _mm256_add_ps(vectorYb, vectorXa);
    _mm256_store_ps(w+i, vectorW);
  }
  // for(; i < n; i++){
  //   w[i] = alpha * x[i] + beta * y[i];
  // }
  return 0;
}
//The loop is very inefficient because the else encapsulates the whole loop essentially