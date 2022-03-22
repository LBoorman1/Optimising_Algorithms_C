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
int ddot (const int n, const float * const x, const float * const y, float * const result) {  
  
  int i;
  int loopFactor = 4; //loopFactor = 4: there are four floats in the 128 bit vector
  int loopN = (n/loopFactor)*loopFactor;

  float local_result = 0.0;

  /*local_result is a reduction variable as result is computed from all of the local results 
  from each thread*/
  #pragma omp parallel for reduction(+:local_result) 
  for (i=0; i<loopN; i+=loopFactor) {
    
    __m128 vectorX = _mm_load_ps(x+i);
    __m128 vectorY = _mm_load_ps(y+i);
    __m128 vectorXY = _mm_mul_ps(vectorX, vectorY);

    //horizontally adds pairs of floats, needs to be done twice as there are four floats in a vector
    __m128 addVec = _mm_hadd_ps(vectorXY,vectorXY);
    __m128 newAddVec = _mm_hadd_ps(addVec, addVec);

    //this takes the high value of the vector, all values will be the same in this case
    local_result+= _mm_cvtss_f32(newAddVec);
  }
  for(i=loopN; i < n; i++){
    local_result += x[i]*y[i];
  }

  *result = local_result;

  return 0;
}
