#include "ddot.h"
#include <omp.h>
#include <immintrin.h>

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  
  
  int i;
  int loopFactor = 2;
  int loopN = (n/loopFactor)*loopFactor;

  //vectorising
  double local_result = 0.0;


  #pragma omp parallel for reduction(+:local_result)
  for (i=0; i<loopN; i+=loopFactor) {
    
    __m128d vectorX = _mm_load_pd(x+i);
    __m128d vectorY = _mm_load_pd(y+i);
    __m128d vectorXY = _mm_mul_pd(vectorX, vectorY);
    __m128d addVec = _mm_hadd_pd(vectorXY,vectorXY);
    local_result+= _mm_cvtsd_f64(addVec);
  }
  if (n%2 == 1){
    local_result += x[n-1]*y[n-1];
  }

  *result = local_result;

  return 0;
}
