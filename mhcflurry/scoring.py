"""
Measures of prediction accuracy
"""
import logging
import sklearn.metrics
import numpy
import scipy

from .regression_target import from_ic50


def make_scores(
        ic50_y,
        ic50_y_pred,
        sample_weight=None,
        threshold_nm=500,
        max_ic50=50000):
    """
    Calculate AUC, F1, and Kendall Tau scores.

    Parameters
    -----------
    ic50_y : float list
        true IC50s (i.e. affinities)

    ic50_y_pred : float list
        predicted IC50s

    sample_weight : float list [optional]

    threshold_nm : float [optional]

    max_ic50 : float [optional]

    Returns
    -----------
    dict with entries "auc", "f1", "tau"
    """
    pass
