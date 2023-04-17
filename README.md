# Gaussian Ansatz (v0.0.1)

[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/rikab/GaussianAnsatz)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7689890.svg)](https://doi.org/10.5281/zenodo.7689890) -->


The `Gaussian Ansatz` is a machine learning framework for performing frequentist inference, complete with local uncertainty estimation, as described in ["Learning Uncertainties the Frequentist Way: Calibration and Correlation in High Energy Physics" (arxiv:2205.03413)](https://arxiv.org/abs/2205.03413). These models can be used to quickly estimate local resolutions of an inference $z$ given a measurement $x$, even if $x$ is very high-dimensional.




## Installation

### From PyPI

In your Python environment run

```
python -m pip install GaussianAnsatz
```

### From this repository locally

In your Python environment from the top level of this repository run

```
python -m pip install .
```

### From GitHub

In your Python environment run

```
python -m pip install "GaussianAnsatz @ git+https://github.com/rikab/GaussianAnsatz.git"
```

## Example Usage

For an example of how to use the Gaussian Ansatz, see the notebook `examples/minimal_working_example.ipynb`. This notebook contains example code for loading data, using pre-built DNN Gaussian Ansatz, pre-training and training, and extracting inferences with uncertainty estimates.

Additional, more complicated examples can be found in the `JEC` subfolder. The files here correspond exactly to the jet energy studies, as described in ["Learning Uncertainties the Frequentist Way: Calibration and Correlation in High Energy Physics" (arxiv:2205.03413)](https://arxiv.org/abs/2205.03413)

## Dependencies

To use the `Gaussian Ansatz`, the following packages must be installed as prerequisites:

- [Tensorflow](https://github.com/tensorflow/tensorflow): A standard tensor operation library.
- [Energyflow](https://energyflow.network/): A suite of particle physics tools, including Energy Flow Networks and Particle FLow Networks
- Standard python packages: [numpy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/)

## Citation

If you use the `Gaussian Ansatz`, please cite both the corresponding paper, "Learning Uncertainties the Frequentist Way: Calibration and Correlation in High Energy Physics"?:



    @article{Gambhir:2022gua,
    author = "Gambhir, Rikab and Nachman, Benjamin and Thaler, Jesse",
    title = "{Learning Uncertainties the Frequentist Way: Calibration and Correlation in High Energy Physics}",
    eprint = "2205.03413",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "MIT-CTP 5431",
    doi = "10.1103/PhysRevLett.129.082001",
    journal = "Phys. Rev. Lett.",
    volume = "129",
    number = "8",
    pages = "082001",
    year = "2022"
}


## Changelog

- v0.0.1: 17 April 2023. Now pip-installable!
- v0.0.0: 6 May 2022. Public release.

Based on the work in ["Learning Uncertainties the Frequentist Way: Calibration and Correlation in High Energy Physics" (arxiv:2205.03413)](https://arxiv.org/abs/2205.03413)

Bugs, Fixes, Ideas, or Questions? Contact me at rikab@mit.edu
