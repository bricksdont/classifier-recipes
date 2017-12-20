# Recipes for sentence classification tasks

Code that exemplifies neural network solutions for classification tasks with DyNet (for the core model) and Keras (preprocessing only). On top of that, the code demonstrates how to implement a custom classifier that is compatible with scikit-learn's API.

## Installation

All examples should be run with Python 2.

Install a recent version of DyNet: http://dynet.readthedocs.io/en/latest/python.html. If you have a CUDA-capable GPU, build with CUDA support, but the models run fine on CPU too.

For the CPU version, you can probably use

    pip install git+https://github.com/clab/dynet#egg=dynet

Other libraries that need to be installed:

    scikit-learn
    keras
    nltk
    pandas

Then clone the repository to your machine.

## Training Models

There are three architecture variants, implemented in `lib/cnn.py`, `lib/rnn.py` and `lib/pooling.py`.

## Example Use Case

As an example, the data for the Spooky Author Identification Challenge (https://www.kaggle.com/c/spooky-author-identification) is included in the `data` folder. CSV files can be read directly, and the predictions are written in a submission-ready format. Adapt the pre- and postprocessing to your needs.

## Compatibility with Scikit-Learn

All classifier classes are compatible with `scikit-learn`'s API. This is achieved by having them inherit from `BaseEstimatorMixin` and `ClassifierMixin`. Compatibility means that they implement the same methods and can be used in cross-validation, pipelines or random search.
