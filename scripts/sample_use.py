# sample from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

from types import MappingProxyType
from typing import Any, List, Mapping, Tuple

import numpy as np
from sklearn import datasets, svm
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

_DEFAULT_SEED: int = 42
_DEFAULT_TEST_SIZE: float = 0.8
_DEFAULT_CLASSIFIER_ARGS = []
_DEFAULT_CLASSIFIED_KWARGS = MappingProxyType(
    {
        "kernel": "linear", 
        "probability": True, 
        "random_state": 42,  
    }
)

def load_iris_data(
    seed: int = _DEFAULT_SEED, test_size: float = _DEFAULT_TEST_SIZE
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load iris dataset for sample classification challenge.

    Args:
        seed: Random seed.
        test_size: Percentage of dataset reserved for test.
    Returns:
        (x_train, y_train), (x_test, y_test) of iris dataset.
    """

    iris = datasets.load_iris()
    x, y = iris.data, iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    return (x_train, y_train), (x_test, y_test)


def get_model(
    classifer_args: List[Any] = _DEFAULT_CLASSIFIER_ARGS,
    classifier_kwargs: Mapping[str, Any] = _DEFAULT_CLASSIFIED_KWARGS,
) -> BaseEstimator:
    """

    Args:
        classifer_args:
        classifier_kwargs:
    """

    return OneVsRestClassifier(
        svm.SVC(*classifer_args, **classifier_kwargs)
    )


def cli_handler(classifier_args, classifier_kwargs):
    ...


if __name__ == "__main__":
    
    (x_train, y_train), (x_test, y_test) = load_iris_data()
    classifier = get_model()

    y_score = classifier.fit(x_train, y_train).decision_function(x_test)


# Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
