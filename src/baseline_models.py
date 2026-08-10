"""Fixed Phase A TF-IDF character n-gram LinearSVC model."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_model(config: dict) -> Pipeline:
    vectorizer = config["vectorizer"]
    classifier = config["classifier"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=vectorizer["analyzer"],
                    ngram_range=tuple(vectorizer["ngram_range"]),
                    min_df=vectorizer["min_df"],
                    sublinear_tf=vectorizer["sublinear_tf"],
                    max_features=vectorizer["max_features"],
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=classifier["C"],
                    class_weight=classifier["class_weight"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    base: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    metric = {
        **base,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            **base,
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusion = [
        {
            **base,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(matrix[row_index, column_index]),
        }
        for row_index, true_label in enumerate(labels)
        for column_index, predicted_label in enumerate(labels)
    ]
    return metric, per_class, confusion
