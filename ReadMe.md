# VRLM: Variance-reduced accelerated methods for decentralized stochastic double-regularized nonconvex strongly-concave minimax problems

This document provides instructions on how to reproduce the experimental results from the [VRLM paper](https://arxiv.org/abs/2307.07113). Covered in this document are:

- package requirements utilized in the experiments
- instructions on how to reproduce the results from the paper (i.e. hyperparameter settings, etc.)
- a description of the main components in this repository, their use, and how to modify them for new use cases

## Experiment set-up

Experiments were ran on clusters of 8 NVIDIA Tesla V100's (each with 32 GiB HBM) connected by dual 100 Gb EDR Infiniband. The operating system utilized is [CentOS](https://www.centos.org) 7 and all experiments are ran within a conda version 4.9.2 environment. All code is written in Python version 3.7.9, using `PyTorch` version 1.6.0 with CUDA version 10.2; for instructions on how to install PyTorch with CUDA see [here](https://pytorch.org/get-started/previous-versions/). The GCC version of the system utilized is 4.8.5. To perform neighbor communication, `mpi4py` was utilized; see [here](https://mpi4py.readthedocs.io/en/stable/install.html) for instructions on how to install this package.

A complete list of packages necessary for completing the experiments is located in the [requirements.txt](setup/requirements.txt) file. Comprehensive installation instructions can be found in [Install.md](setup/Install.md).


## Running experiments


## Repository summary

This repository contains the following directories and files:
```
mixing_matrices/
DRO/
   models/
   main.py
FC/
   models/
   main.py
requirements.txt
```

#### mixing_matrices
The `mixing_matrices` folder contains `Numpy` arrays of size `N x N` where each `(i,j)` entry corresponds to agent `i`'s weighting of agent `j`'s information

#### main.py

As stated above, the `main.py` scripts actually perform the training. **If you add implement a new method, do not forget to edit this file accordingly**

#### models
The `models` folder(s) contain the bulk of the code required to reproduce the experiments from the paper. To add a new problem instance or neural network architecture, include said file here. To add a new method, create a file in the same directory as the `base.py` file and implement a class with the following general structure:

```
class NewMethod(BaseDL):
    def __init__(self,
                 params: typing.Dict,
                 mixing_matrix: numpy.array,
                 training_data: torch.utils.data.DataLoader,
                 mega_batch_train_loader: torch.utils.data.DataLoader,
                 full_batch_train_loader: torch.utils.data.DataLoader,
                 comm_world,
                 comm_size,
                 current_rank
                 ):

        super().__init__(params,
                 mixing_matrix,
                 training_data,
                 mega_batch_train_loader,
                 full_batch_train_loader,
                 comm_world,
                 comm_size,
                 current_rank,
                 "<new_method_name>",
                 f"<new_method_save_string>")

        # Extract parameters if you need to introduce any new parameters to your algorithm
        if 'param1' in params:
            self.param1 = params['param1']
        else:
            self.param1 = <some_default_value>

    def one_step(self, iteration: int) -> tuple:

        # Implement one step of your algorithm - MAKE SURE TO TIME THE COMPUTATION AND COMMUNICATION COST

        return comp_time, comm_time
```

## Citation

If you use this code in your research, please cite our paper:
```
@article{mancino2023variance,
  title={Variance-reduced accelerated methods for decentralized stochastic double-regularized nonconvex strongly-concave minimax problems},
  author={Mancino-Ball, Gabriel and Xu, Yangyang},
  journal={arXiv preprint arXiv:2307.07113},
  year={2023}
}
```



