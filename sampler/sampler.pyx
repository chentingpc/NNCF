# distutils: language = c++
# distutils: sources = nodesampler.cpp
from libc.stdlib cimport malloc, free


import cython

import numpy as np
cimport numpy as np

cdef extern from "nodesampler.cpp" namespace "nodesampler":
  cdef cppclass NodeSampler:
    NodeSampler(double *, int, double, unsigned long long) except +
    double *dist
    int dist_size
    int sample()
    void sample_batch(int, int *)

cdef class MultinomialSampler:
  """
  sample from given categorical distribution
  Note: sample batch size: int, sample node index: int
  """
  cdef NodeSampler *thisptr      # hold a C++ instance which we're wrapping
  def __cinit__(self, np.ndarray[double, ndim=1, mode="c"] input not None, int dist_size,
      double neg_sampling_power = 0.75, unsigned long long rand_seed = 0):
    self.thisptr = new NodeSampler(&input[0], dist_size, neg_sampling_power, rand_seed)
  def __dealloc__(self):
    del self.thisptr
  def sample(self):
    # return a node index (int)
    return self.thisptr.sample()
  def sample_batch(self, int n):
    # return an array of node index (int)
    # implementation 1: faster when return one big np.array
    cdef np.ndarray[int, ndim=1, mode="c"] result = np.zeros(n, dtype=np.int32)
    self.thisptr.sample_batch(n, &result[0])
    return result
    # implementation 2: faster when return lots list
    #cdef int *result = <int *> malloc(n * sizeof(int))
    #if not result:
    #  raise MemoryError()
    #try:
    #  self.thisptr.sample_batch(n, result)
    #  result_safe = [result[i] for i in range(n)]
    #finally:
    #  free(result)
    #return result_safe
