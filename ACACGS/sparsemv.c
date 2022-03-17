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
int sparsemv(struct mesh *A, const double * const x, double * const y)
{

  const int nrow = (const int) A->local_nrow; //number of rows in the A matrix

  #pragma omp for
  for (int i=0; i< nrow; i++) {
      double sum = 0.0; //needs to be inside loop so it is set to 0 for every row
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i]; //array of non-zero values 
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i]; //array of indexes where the non-zero values are
      const int cur_nnz = (const int) A->nnz_in_row[i]; //number of non-zeros in the row

      //#pragma omp parallel for reduction (+:sum)
      for (int j=0; j< cur_nnz; j++) {
        sum += cur_vals[j]*x[cur_inds[j]];
      }
      y[i] = sum;
    }
  return 0;
}
