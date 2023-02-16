# Features Maximization Metric

[![ci](https://github.com/cognitivefactory/features-maximization-metric/workflows/ci/badge.svg)](https://github.com/cognitivefactory/features-maximization-metric/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://cognitivefactory.github.io/features-maximization-metric/)
[![pypi version](https://img.shields.io/pypi/v/cognitivefactory-features-maximization-metric.svg)](https://pypi.org/project/cognitivefactory-features-maximization-metric/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7646382.svg)](https://doi.org/10.5281/zenodo.7646382)

Implementation of _Features Maximization Metric_, an unbiased metric aimed at estimate the quality of an unsupervised classification.


## <a name="Description"></a> Quick description

_Features Maximization_ (`FMC`) is a features selection method described in `Lamirel J.-C., Cuxac P., Hajlaoui K., A new approach for feature selection based on quality metric, Advances in Knowledge Discovery and Management, 6 (665), Springer.`

This metric is computed by applying the following steps:

1. Compute the ***Features F-Measure*** metric (based on ***Features Recall*** and ***Features Predominance*** metrics).

    > (a) The ***Features Recall*** `FR[f][c]` for a given class `c` and a given feature `f` is the ratio between
    > the sum of the vectors weights of the feature `f` for data in class `c`
    > and the sum of all vectors weights of feature `f` for all data.
    > It answers the question: "_Can the feature `f` distinguish the class `c` from other classes `c'` ?_"

    > (b) The ***Features Predominance*** `FP[f][c]` for a given class `c` and a given feature `f` is the ratio between
    > the sum of the vectors weights of the feature `f` for data in class `c`
    > and the sum of all vectors weights of all feature `f'` for data in class `c`.
    > It answers the question: "_Can the feature `f` better identify the class `c` than the other features `f'` ?_"

    > (c) The ***Features F-Measure*** `FM[f][c]` for a given class `c` and a given feature `f` is
    > the harmonic mean of the ***Features Recall*** (a) and the ***Features Predominance*** (c).
    > It answers the question: "_How much information does the feature `f` contain about the class `c` ?_"

2. Compute the ***Features Selection*** (based on ***F-Measure Overall Average*** comparison).

    > (d) The ***F-Measure Overall Average*** is the average of ***Features F-Measure*** (c) for all classes `c` and for all features `f`.
    > It answers the question: "_What are the mean of information contained by features in all classes ?_"

    > (e) A feature `f` is ***Selected*** if and only if it exist at least one class `c` for which the ***Features F-Measure*** (c) `FM[f][c]` is bigger than the ***F-Measure Overall Average*** (d).
    > It answers the question: "_What are the features which contain more information than the mean of information in the dataset ?_"

    > (f) A Feature `f` is ***Deleted*** if and only if the ***Features F-Measure*** (c) `FM[f][c]` is always lower than the ***F-Measure Overall Average*** (d) for each class `c`.
    > It answers the question: "_What are the features which do not contain more information than the mean of information in the dataset ?_"

3. Compute the ***Features Contrast*** and ***Features Activation*** (based on ***F-Measure Marginal Averages*** comparison).

    > (g) The ***F-Measure Marginal Averages*** for a given feature `f` is the average of ***Features F-Measure*** (c) for all classes `c` and for the given feature `f`.
    > It answers the question: "_What are the mean of information contained by the feature `f` in all classes ?_"

    > (h) The ***Features Contrast*** `FC[f][c]` for a given class `c` and a given selected feature `f` is the ratio between
    > the ***Features F-Measure*** (c) `FM[f][c]`
    > and the ***F-Measure Marginal Averages*** (g) for selected feature f
    > put to the power of an ***Amplification Factor***.
    > It answers the question: "_How relevant is the feature `f` to distinguish the class `c` ?_"

    > (i) A selected Feature `f` is ***Active*** for a given class `c` if and only if the ***Features Contrast*** (h) `FC[f][c]` is bigger than `1.0`.
    > It answers the question : "_For which classes a selected feature `f` is relevant ?_"

This metric is an **efficient method** to:

- **identify relevant features** of a dataset modelization;
- **describe association** between vectors features and data classes;
- **increase contrast** between data classes.


## <a name="Documentation"></a> Documentation

- [Main documentation](https://cognitivefactory.github.io/features-maximization-metric/)


## <a name="Installation"></a> Installation

Features Maximization Metric requires [`Python`](https://www.python.org/) 3.8 or above.

To install with [`pip`](https://github.com/pypa/pip):

```bash
# install package
python3 -m pip install cognitivefactory-features-maximization-metric
```

To install with [`pipx`](https://github.com/pypa/pipx):

```bash
# install pipx
python3 -m pip install --user pipx

# install package
pipx install --python python3 cognitivefactory-features-maximization-metric
```


## <a name="Development"></a> Development

To work on this project or contribute to it, please read:

- the [Copier PDM](https://pawamoy.github.io/copier-pdm/) template documentation ;
- the [Contributing](https://cognitivefactory.github.io/features-maximization-metric/contributing/) page for environment setup and development help ;
- the [Code of Conduct](https://cognitivefactory.github.io/features-maximization-metric/code_of_conduct/) page for contribution rules.


## <a name="References"></a> References

- **Features Maximization Metric**: `Lamirel J.-C., Cuxac P., Hajlaoui K., A new approach for feature selection based on quality metric, Advances in Knowledge Discovery and Management, 6 (665), Springer.`
- **V-Measure**: `Rosenberg, Andrew & Hirschberg, Julia. (2007). V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure. 410-420.`


## <a name="How to cite"></a> How to cite	

`Schild, E. (2023). cognitivefactory/features-maximization-metric. Zenodo. https://doi.org/10.5281/zenodo.7646382.`