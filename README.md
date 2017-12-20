# Recipes for sentence classification tasks

Code that exemplifies neural network solutions for classification tasks with `DyNet` (for the core model) and `Keras` (preprocessing only). On top of that, the code demonstrates how to implement a custom classifier that is compatible with scikit-learn's API.

## Installation

All examples should be run with Python 2. The following libraries need to be installed: `scikit-learn`, `keras`, `nltk`,  `pandas` and `DyNet`.

Setting up a virtual Python environment is a good idea, for instance with `conda`. Then, inside your environment,

    conda install scikit-learn keras nltk pandas

Install a recent version of DyNet: http://dynet.readthedocs.io/en/latest/python.html. If you have a CUDA-capable GPU, build with CUDA support, but the models run fine on CPU too.

For the CPU version, you can probably use (from within your virtual environment)

    pip install git+https://github.com/clab/dynet#egg=dynet

Then clone the repository to your machine.

## Training Models

There are three architecture variants, implemented in `lib/cnn.py`, `lib/rnn.py` and `lib/pooling.py`. Train a model with, e.g.:

    python lib/pooling.py --train_file [training CSV] --early_stop

Use `-h/--help` for more command line parameters:

    python pooling.py --help

## Example Use Case

As an example, the data for the Spooky Author Identification Challenge (https://www.kaggle.com/c/spooky-author-identification) is included in the `data` folder. CSV files can be read directly, and the predictions are written in a submission-ready format:

    python lib/pooling.py --train_file data/train.csv --test_file data/test.csv --early_stop

Adapt the pre- and postprocessing (in `lib/preprocessing.py`) to your needs.

## Compatibility with Scikit-Learn

The `pooling.PoolingClassifier` and `rnn.RNNClassifier` classes are compatible with `scikit-learn`'s API. This is achieved by having them inherit from `BaseEstimatorMixin` and `ClassifierMixin`. Compatibility means that they implement the same methods and can be used in cross-validation, pipelines or random search.

For instance, use them directly with `scikit-learn` toy datasets:

```python
from sklearn.datasets import load_iris
import pooling

p = pooling.PoolingClassifier()
p # in an interpreter, prints all default parameters and their values
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
p.fit(X, y)
```
