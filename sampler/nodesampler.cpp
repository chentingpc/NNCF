#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <ctime>

namespace nodesampler {
  typedef float real;                     // Precision of float numbers
  typedef unsigned int uint;
  typedef long long int64;
  typedef unsigned long long uint64;

  #define NEG_SAMPLING_POWER 0.75         // unigram downweighted
  const int64 neg_table_size = 1e8;

  class NodeSampler {
    int                 *neg_table;         // based on overall network
    int                 num_vertices;
    uint64              seed;
    double              neg_sampling_power;

    /* Fastly generate a random integer */
    inline int Rand(uint64 &seed) {
      seed = seed * 25214903917 + 11;
      return (seed >> 16) % neg_table_size;
    }

    // zero degree node will not be sampled
    void set_table(int *table, const double *vertex_degree) {
      double sum = 0, cur_sum = 0, por = 0, deg;
      int k = 0;
      for (int i = 0; i != num_vertices; i++) {
          deg = vertex_degree[i];
          if (deg == 0.) continue;
          sum += pow(vertex_degree[i], neg_sampling_power);
      }
      for (int i = 0; i < num_vertices; i++) {
          deg = vertex_degree[i];
          if (deg == 0.) continue;
          cur_sum += pow(deg, neg_sampling_power);
          por = cur_sum / sum;
          while ((double)k / neg_table_size < por && k != neg_table_size) {
            table[k++] = i;
          }
      }
      if (k != neg_table_size)
        printf("%d, %lld\n", k, neg_table_size);
      assert(k == neg_table_size);  // even not hold, they should be close, check precision
    }

   public:

    // build a sampler from a discrete distribution of size dist_size
    // should only use sample(seed) for sample
    explicit NodeSampler(const double *dist, const int dist_size,
        const double neg_sampling_power = NEG_SAMPLING_POWER, const uint64 rand_seed = 0):
        neg_table(NULL) {
      this->neg_sampling_power = neg_sampling_power;
      if (rand_seed == 0)
        this->seed = time(NULL);
      num_vertices = dist_size;
      neg_table = new int[neg_table_size];
      if (dist_size > neg_table_size * 0.1)
        printf("[Warning!] dist_size is larger than neg_table_size * 0.1, should consider increase neg_table_size\n");
      assert(dist_size < neg_table_size);
      set_table(neg_table, dist);
    }

    inline int sample() {
      return neg_table[Rand(seed)];
    }

    inline void sample_batch(int n, int *result) {
      for (size_t i = 0; i < n; i++)
        result[i] = neg_table[Rand(seed)];
    }
  };
}  // namespace nodesampler
