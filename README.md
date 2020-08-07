# Bayesian Optimization with Missing Inputs

The implementation of BOMI in the paper 'Bayesian Optimization with Missing Inputs', ECMLPKDD2020.

## Prerequisites

- Python 3.6
- Numpy 1.18
- Scipy 1.3.1
- Scikit-learn 0.21.2
- Torch 1.3.1 (CUDA v9.2)
- Gpytorch 1.0.1
- Missingpy 0.2.0
- Pandas 0.25.3

(Optional)
- pip 19.3.1
- pillow 5.4.1

## Instruction

- Execute an experiment with the command: ..\python runExperiment <opt_method> <obj_function> <num_of_GPs> <alpha_param> <miss_rate> <miss_noise>

Example: 
  ````md
  python runExperiemnt BOMI Eggholder2d 5 1e2 0.25 0.05
  ````
  
See 'runExperiment.py' for more details and see 'ndfunction.py' for how to define a new objective function.

## Reference

- Paper [Bayesian Optimization with Missing Inputs](https://arxiv.org/pdf/2006.10948.pdf)

## License

Apache 2.0
