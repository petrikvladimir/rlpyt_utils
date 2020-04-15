# Introduction
Some utilities for managing experiments in rlpyt framework, e.g. to automatically increase run_id to not overwrite previous run under the same name.


# Utils
## Exponential reward

Computes unit reward for vector. Parameter _b_ is used to specify lengthscale of the reward according to the table:

exp | d=0.01 | d=0.05 | d=0.1 | d=0.5 | d=1.0 | d=5.0 | d=10.0
--- | --- | --- | --- | --- | --- | --- | ---
**b=1e-02** | 1.0e+00 | 1.0e+00 | 1.0e+00 | 1.0e+00 | 1.0e+00 | 8.8e-01 | 6.1e-01
**b=1e-01** | 1.0e+00 | 1.0e+00 | 1.0e+00 | 9.9e-01 | 9.5e-01 | 2.9e-01 | 6.7e-03
**b=1e+01** | 1.0e+00 | 9.9e-01 | 9.5e-01 | 2.9e-01 | 6.7e-03 | 5.2e-55 | 7.1e-218
**b=1e+02** | 1.0e+00 | 8.8e-01 | 6.1e-01 | 3.7e-06 | 1.9e-22 | 0.0e+00 | 0.0e+00
**b=1e+03** | 9.5e-01 | 2.9e-01 | 6.7e-03 | 5.2e-55 | 7.1e-218 | 0.0e+00 | 0.0e+00
**b=1e+04** | 6.1e-01 | 3.7e-06 | 1.9e-22 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00