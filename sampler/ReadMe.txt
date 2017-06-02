This module implements some samplers for python program.

sampler_python: multinomial sampler implemented in python, real slow.

sampler: multinomial sampler implemented in C++, ported by cython, real fast. A single sample can be 9x faster than sampler_python. With batch sampling, it can easy be 10x or even 100x faster than sampler_python.



