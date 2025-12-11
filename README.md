# Differentially-Private Federated Learning with Personalisation

## Description
Project for evaluating federated learning with central or local differential privacy and an optional fine-tuning step for model personalisation.
This code was used for our paper _Pfitzner, B., Maurer, M. M., Winter, A., Riepe, C., Sauer, I. M., Van de Water, R., & Arnrich, B. (2024, February). Differentially-Private Federated Learning with Non-IID Data for Surgical Risk Prediction. In 2024 IEEE First International Conference on Artificial Intelligence for Medicine, Health and Care (AIMHC) (pp. 120-129). IEEE._

## Installation
First, create an environment with conda (or any other environment management software you prefer) using Python 3.8 and activate it:

```bash
$ conda create --name dp_fl_per python=3.8.0
$ conda activate dp_fl_per
```

Then, install the required packages from the `requirements.txt` file using `pip`:

```bash
$ pip install -r requirements.txt
```

In order to run the code with the example dataset [Support2](http://archive.ics.uci.edu/dataset/880/support2) from the UCI ML repository, you need to also install their package:

```bash
$ pip install ucimlrepo
```

## Usage
Experiments are configured with the [Hydra](https://hydra.cc) framework using the yaml files in the top-level [config](config) folder.
[config.yaml](config%2Fconfig.yaml) holds all the base configurations and the subfolders (except for [experiment](config%2Fexperiment)) hold templates for different configurations (e.g. the [training](config%2Ftraining) folder holds the base configurations for local, centralised and federated learning).
Specific overrides for an experiment are aggregated in the yaml files inside the [experiment](config%2Fexperiment) folder.
These are selected in the python command to run, like so:

```bash
$ python -m src.main +experiment=federated_support2_cdp
```

Please refer to [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) for an explanation on how to change specific configs from the command line.

The configurations used for the abovementioned paper can be found in [experiment/paper](config%2Fexperiment%2Fpaper), but since the data cannot be made public for privacy reasons, they are just for reference.

We use [Weights and Biases](https://wandb.ai) for experiment tracking, so, if you set up your API key on your system, it should automatically upload the experiment results to a new project.
However, we implemented additional logging to the output folder specified by the config `output_folder`, where we will store the trained model as well as a `metrics.csv` with all recorded metrics over all steps.

For running a hyperoptimisation, we use the [hydra-wandb-sweeper](https://github.com/captain-pool/hydra-wandb-sweeper) that allowed us to track the runs directly to WandB. 
You can adapt the `hydra.sweeper` configuration in [config.yaml](config%2Fconfig.yaml) to be suitable for your WandB setup, or change it to a different sweeper entirely.
We configure the search parameters in the experiment files, as shown in [sample_hyperparam_search_centralised_death.yaml](config%2Fexperiment%2Fsample_hyperparam_search_centralised_death.yaml).

## Support
If there are any questions or issues, please contact me at [bjarne.pfitzner@hpi.de](mailto:bjarne.pfitzner@hpi.de?subject=[GitHub]).
