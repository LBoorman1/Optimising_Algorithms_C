#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>


#include "sparsemv.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */

// |From 256 bit method - not being used but thought it was interesting|
// float sum256vector(__m256d vector) {
//   __m128d lowhalf = _mm256_castpd256_pd128(vector);
//   __m128d highhalf = _mm256_extractf128_pd(vector, 1);
//   lowhalf = _mm_add_pd(lowhalf, highhalf);
//   __m128d hightest = _mm_unpackhi_pd(lowhalf, lowhalf);
//   __m128d vectorAdd = _mm_add_sd(lowhalf, hightest); 
//   return _mm_cvtsd_f64(vectorAdd);
// }

int sparsemv(struct mesh *A, const float * const x, float * const y)
{

  const int nrow = (const int) A->local_nrow; //number of rows in the A matrix

  #pragma omp parallel for
  for (int i=0; i< nrow; i++) {
      float sum = 0.0; //needs to be inside loop so it is set to 0 for every row
      const float * const cur_vals = (const float * const) A->ptr_to_vals_in_row[i]; //array of non-zero values 
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i]; //array of indexes where the non-zero values are
      const int cur_nnz = (const int) A->nnz_in_row[i]; //number of non-zeros in the row - works as n for inner loop

      int loopFactor = 4; //four floats in the vectors
      int loopN = (cur_nnz/loopFactor)*loopFactor;

      //#pragma omp parallel for reduction (+:sum)
      for (int j=0; j<loopN; j+=loopFactor) {

        // |From 256 bit method - not being used but thought it was interesting|
        // __m256d vectorValues = _mm256_load_pd(cur_vals + j);
        // __m256d vectorXVals = _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]);
        // __m256d vectorMul = _mm256_mul_pd(vectorValues, vectorXVals);
        // float scalarAdd = sum256vector(vectorMul);
        // sum += scalarAdd;

        __m128 vectorVals = _mm_loadu_ps(cur_vals + j);
        __m128 vectorX = _mm_set_ps(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]);
        __m128 vectorValsX = _mm_mul_ps(vectorVals, vectorX);

        //horizontally adds pairs of floats, needs to be done twice as there are four floats in a vector
        __m128 vectorAdd = _mm_hadd_ps(vectorValsX, vectorValsX);
        __m128 vectorNewAdd = _mm_hadd_ps(vectorAdd, vectorAdd);

        //this takes the high value of the vector, all values will be the same in this case
        sum+=_mm_cvtss_f32(vectorNewAdd);

      }
      for(int j=loopN; j<cur_nnz; j++){
        sum+= cur_vals[loopN]*x[cur_inds[j]];
      }
      y[i] = sum;
    }
  return 0;
}
