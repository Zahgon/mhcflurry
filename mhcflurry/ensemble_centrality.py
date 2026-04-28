"""
Measures of centrality (e.g. mean) used to combine predictions across an
ensemble. The input to these functions are log affinities, and they are expected
to return a centrality measure also in log-space.
"""

import numpy


def _nanmean_no_warnings(log_values):
    """
    Row-wise nanmean that returns nan for all-nan rows without warnings.
    """
    pass


def _nanmedian_no_warnings(log_values):
    """
    Row-wise nanmedian that returns nan for all-nan rows without warnings.
    """
    pass


def robust_mean(log_values):
    """
    Mean of values falling within the 25-75 percentiles.

    Parameters
    ----------
    log_values : 2-d numpy.array
        Center is computed along the second axis (i.e. per row).

    Returns
    -------
    center : numpy.array of length log_values.shape[1]

    """
    pass


CENTRALITY_MEASURES = {
    "mean": _nanmean_no_warnings,
    "median": _nanmedian_no_warnings,
    "robust_mean": robust_mean,
}
