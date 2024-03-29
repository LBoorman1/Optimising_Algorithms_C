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
  
  /*The whole loop is re-written into one condition that always gets hit.
  The point here was to decrease the complexity of the code before trying
  to make any other optimisations*/

  int i;
  int loopFactor = 4; //loop factor = 4: as there will be four floats in each vector
  int loopN = (n/loopFactor)*loopFactor;
  
  //set vectors for the scalar values
  __m128 vectorB = _mm_set1_ps(beta); 
  __m128 vectorA = _mm_set1_ps(alpha);

  #pragma omp parallel for
  for(i=0; i < loopN; i += loopFactor){
    __m128 vectorX = _mm_load_ps(x+i);
    __m128 vectorY = _mm_load_ps(y+i);
    __m128 vectorXa = _mm_mul_ps(vectorX, vectorA);
    __m128 vectorYb = _mm_mul_ps(vectorY, vectorB);
    __m128 vectorW = _mm_add_ps(vectorYb, vectorXa);
    _mm_store_ps(w+i, vectorW);
  }
  //loop cleanup
  for(i=loopN; i < n; i++){
    w[i] = alpha * x[i] + beta * y[i];
  }
  return 0;
}
