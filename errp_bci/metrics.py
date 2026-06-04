"""Metrics — primary: balanced accuracy and AUC-ROC (ported verbatim)."""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)


def compute_comprehensive_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute all metrics. Primary: balanced_accuracy, auroc. Secondary: accuracy.

    Args:
        predictions  : Predicted labels (0 or 1).
        labels       : Ground-truth labels (0 or 1).
        probabilities: Predicted probabilities shape (N,2) or (N,). Optional for AUC.

    Returns:
        Dict with balanced_accuracy, auroc, accuracy, f1_score, confusion matrix.
    """
    predictions = np.asarray(predictions).ravel()
    labels      = np.asarray(labels).ravel()

    metrics: Dict[str, Any] = {
        'accuracy'          : float(accuracy_score(labels, predictions)),
        'balanced_accuracy' : float(balanced_accuracy_score(labels, predictions)),
        'f1_score'          : float(f1_score(labels, predictions,
                                             average='binary', zero_division=0.0)),
    }

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                    'n_samples': len(labels),
                    'n_class_0': int(np.sum(labels == 0)),
                    'n_class_1': int(np.sum(labels == 1))})

    if probabilities is not None:
        probs = np.asarray(probabilities)
        p1 = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()
        try:
            metrics['auroc'] = float(roc_auc_score(labels, p1))
        except ValueError:
            metrics['auroc'] = float('nan')
    else:
        metrics['auroc'] = float('nan')

    return metrics


def compute_chance_level(labels: np.ndarray) -> Dict[str, float]:
    """Chance level = always predict majority class. Balanced acc = 0.5 by definition."""
    majority = int(np.bincount(labels).argmax())
    preds = np.full_like(labels, majority)
    return compute_comprehensive_metrics(preds, labels)
