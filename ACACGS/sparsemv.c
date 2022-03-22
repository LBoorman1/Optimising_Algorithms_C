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

double sum256vector(__m256d vector) {
  __m128d lowhalf = _mm256_castpd256_pd128(vector);
  __m128d highhalf = _mm256_extractf128_pd(vector, 1);
  lowhalf = _mm_add_pd(lowhalf, highhalf);
  __m128d hightest = _mm_unpackhi_pd(lowhalf, lowhalf);
  __m128d vectorAdd = _mm_add_sd(lowhalf, hightest); 
  return _mm_cvtsd_f64(vectorAdd);
}

//Basically doing matrix multiplication A*x
int sparsemv(struct mesh *A, const double * const x, double * const y)
{

  const int nrow = (const int) A->local_nrow; //number of rows in the A matrix

  #pragma omp parallel for
  for (int i=0; i< nrow; i++) {
      double sum = 0.0; //needs to be inside loop so it is set to 0 for every row
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i]; //array of non-zero values 
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i]; //array of indexes where the non-zero values are
      const int cur_nnz = (const int) A->nnz_in_row[i]; //number of non-zeros in the row - works as n for inner loop

      int loopFactor = 4;
      int loopN = (cur_nnz/loopFactor)*loopFactor;

      //#pragma omp parallel for reduction (+:sum)
      for (int j=0; j<loopN; j+=loopFactor) {
        
        __m256d vectorValues = _mm256_load_pd(cur_vals + j);
        __m256d vectorXVals = _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]);
        __m256d vectorMul = _mm256_mul_pd(vectorValues, vectorXVals);
        double scalarAdd = sum256vector(vectorMul);

        sum += scalarAdd;
      }
      for(int j=loopN; j<cur_nnz; j++){
        sum+= cur_vals[loopN]*x[cur_inds[j]];
      }
      y[i] = sum;
    }
  return 0;
}
