# VarDiU: A Variational Diffusive Upper Bound for One-Step Diffusion Distillation

[![Paper](https://img.shields.io/badge/paper-arXiv:2508.20646-B31B1B.svg)](https://arxiv.org/abs/2508.20646)


## Abstract
Recently, diffusion distillation methods have compressed thousand-step teacher diffusion models into one-step student generators while preserving sample quality. Most existing approaches train the student model using a diffusive divergence whose gradient is approximated via the student's score function, learned through denoising score matching (DSM). Since DSM training is imperfect, the resulting gradient estimate is inevitably biased, leading to sub-optimal performance. In this paper, we propose VarDiU (pronounced /va:rdju:/), a Variational Diffusive Upper Bound that admits an unbiased gradient estimator and can be directly applied to diffusion distillation. Using this objective, we compare our method with Diff-Instruct and demonstrate that it achieves higher generation quality and enables a more efficient and stable training procedure for one-step diffusion distillation.


## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* You have a `Windows/Linux/Mac` machine.

## Installation

To install the necessary packages and set up the environment, follow these steps:

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Leyangw/VarDiU.git
cd VarDiU
```

### Create the Conda Environment

To create the Conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yaml
```

## Usage


### Training
After activating the environment, you can run the scripts or use the modules provided in the repository. 
```bash
conda activate VarDiU
```

Train VarDiU Gaussian variational disribution with true score given:

```bash
python train_upper_diffusion.py -true_score  --sigma_min 0.1 --sigma_max 20 --power 2.0 --device cuda:0 --weight sigma2 --seed 0
```

Train VarDiU Flow variational disribution with true score given:

```bash
python train_upper_flow.py -true_score  --sigma_min 0.1 --sigma_max 20 --power 2.0 --device cuda:0 --weight sigma2 --seed 0 --flow_type NSF --flow_length 4
```


## Citation

If you find our paper, code, and/or data useful for your research, please cite our paper:

```
@article{wang2025vardiu,
  title={VarDiU: A Variational Diffusive Upper Bound for One-Step Diffusion Distillation},
  author={Wang, Leyang and Zhang, Mingtian and Ou, Zijing and Barber, David},
  journal={arXiv preprint arXiv:2508.20646},
  year={2025}
}
```