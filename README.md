# VarDiU: A Variational Diffusive Upper Bound for One-Step Diffusion Distillation

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
After activating the environment, you can run the scripts or use the modules provided in the repository. Train VarDiU Gaussian variational disribution with true score given:

```bash
python train_upper_diffusion.py -true_score  --sigma_min 0.1 --sigma_max 20 --power 2.0 --device cuda:0 --weight sigma2 --seed 0
```

Train VarDiU Flow variational disribution with true score given:

```bash
python train_upper_flow.py -true_score  --sigma_min 0.1 --sigma_max 20 --power 2.0 --device cuda:0 --weight sigma2 --seed 0 --flow_type NSF --flow_length 4
``