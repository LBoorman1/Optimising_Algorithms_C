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


//Basically doing matrix multiplication A*x
int sparsemv(struct mesh *A, const float * const x, float * const y)
{

  const int nrow = (const int) A->local_nrow; //number of rows in the A matrix


  for (int i=0; i< nrow; i++) {
      float sum = 0.0; //needs to be inside loop so it is set to 0 for every row
      const float * const cur_vals = (const float * const) A->ptr_to_vals_in_row[i]; //array of non-zero values 
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i]; //array of indexes where the non-zero values are
      const int cur_nnz = (const int) A->nnz_in_row[i]; //number of non-zeros in the row


      int loopFactor = 2;
      int loopN = (cur_nnz/loopFactor)*loopFactor;

      #pragma omp parallel for
      for (int j=0; j<loopN; j+=loopFactor){
        __m128d vectorVals = _mm_loadu_ps(cur_vals + j); //get cur_vals into a vector

        __m128d vectorX = _mm_set_ps(x[cur_inds[j+1]], x[cur_inds[j]]); //get two of the x indexes into a vector

        __m128d vectorValsX = _mm_mul_ps(vectorVals, vectorX); //start arithmetic

        __m128d vectorAdd = _mm_hadd_ps(vectorValsX, vectorValsX); //add pairs of values in vectors

        sum += _mm_cvtsd_f64(vectorAdd);
      }
      //finish iteration incase loop number was uneven
      for(int j=loopN; j<cur_nnz; j++) {
        sum+=cur_vals[loopN]*x[cur_inds[j]];
      }

      y[i] = sum;

    }
  return 0;
}
