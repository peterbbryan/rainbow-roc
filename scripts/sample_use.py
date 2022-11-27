"""
Rainbow ROC demo of visualized thresholds.

Derived from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
"""

from types import MappingProxyType
from typing import Any, Mapping, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

_DEFAULT_SEED: int = 42
_DEFAULT_TEST_SIZE: float = 0.8
_DEFAULT_CLASSIFIER_ARGS = ()
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
    x, y = iris.data, iris.target  # pylint: disable=no-member
    y = label_binarize(y, classes=[0, 1, 2])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed
    )

    return (x_train, y_train), (x_test, y_test)


def get_model(
    classifer_args: Tuple,
    classifier_kwargs: Mapping[str, Any],
) -> BaseEstimator:
    """
    Get one v. rest SVC classifier.

    Args:
        classifer_args: SVM arguments.
        classifier_kwargs: SVM key word arguments.
    """

    return OneVsRestClassifier(svm.SVC(*classifer_args, **classifier_kwargs))


def cli_handler(
    seed: int = _DEFAULT_SEED,
    test_size: float = _DEFAULT_TEST_SIZE,
    classifier_args: Tuple = _DEFAULT_CLASSIFIER_ARGS,
    classifier_kwargs: Mapping[str, Any] = _DEFAULT_CLASSIFIED_KWARGS,
    output_file: str = "rainbow_curve.png",
    labels: Tuple = ("Setosa", "Versicolour", "Virginica"),
) -> None:
    """
    CLI handler for plotting.

    Args:
        seed: Random seed.
        test_size: Percentage of dataset reserved for test.
        classifer_args: SVM arguments.
        classifier_kwargs: SVM key word arguments.
    """

    (x_train, y_train), (x_test, y_test) = load_iris_data(
        seed=seed, test_size=test_size
    )
    classifier = get_model(
        classifer_args=classifier_args, classifier_kwargs=classifier_kwargs
    )

    y_score = classifier.fit(x_train, y_train).decision_function(x_test)

    categorical_labels = np.argmax(np.concatenate((y_train, y_test)), axis=-1)
    unique_labels = np.sort(np.unique(categorical_labels))

    plt.figure(figsize=(5, 12))

    for i in unique_labels:
        fpr, tpr, thresholds = roc_curve(y_test[:, i], y_score[:, i])
        plt.subplot(len(unique_labels), 1, i + 1)
        plt.plot(fpr, tpr, c="black")
        plt.scatter(fpr, tpr, c=thresholds, cmap="rainbow")
        plt.axis("square")
        plt.colorbar()
        plt.title(labels[i])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

    plt.tight_layout()
    plt.savefig(output_file)


if __name__ == "__main__":
    fire.Fire(cli_handler)
