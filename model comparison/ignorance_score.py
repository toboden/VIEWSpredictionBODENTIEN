import numpy as np
from collections import Counter

bins = [
        0,
        0.5,
        2.5,
        5.5,
        10.5,
        25.5,
        50.5,
        100.5,
        250.5,
        500.5,
        1000.5,
    ]

def _ensemble_ignorance_score(predictions, n, observed):
        c = Counter(predictions)
        # n = c.total() : this works from python version 3.10, avoid this for a while.
        prob = c[observed] / n # if counter[observed] is 0, then this returns correctly
        return -np.log2(prob)

def ensemble_ignorance_score(observations, forecasts, bins, low_bin = 0, high_bin = 10000):
    """
    This implements the Ensemble (Ranked) interval Score from the easyVerification R-package in Python. Also inspired by properscoring.crps_ensemble(),
    and has interface that works with the xskillscore package.

    Parameters
    ----------
    observations : float or array_like
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    forecasts : float or array_like
        Array of forecasts ensemble members, of the same shape as observations
        except for the axis along which RIGN is calculated (which should be the
        axis corresponding to the ensemble). If forecasts has the same shape as
        observations, the forecasts are treated as deterministic. Missing
        values (NaN) are ignored.
    round_values: converts input data to integers by rounding.
    

    Returns
    -------
    out : np.ndarray
        RIGN for each ensemble forecast against the observations.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)

    assert np.all(forecasts >= 0), f"Forecasts must be non-negative."
    assert np.all(observations >= 0), f"Observations must be non-negative."

    assert isinstance(bins, (int, list)), f"bins must be an integer or a list with floats"
    if isinstance(bins, int):
        assert bins > 0, f"bins must be an integer above 0 or a list with floats."

    def digitize_minus_one(x, bins, right=False):
        return np.digitize(x, bins, right) - 1

    """ edges = np.histogram_bin_edges(forecasts[..., :], bins = bins, range = (low_bin, high_bin))
    binned_forecasts =  np.apply_along_axis(digitize_minus_one, axis = 1, arr = forecasts, bins = edges)
    binned_observations = digitize_minus_one(observations, edges) """

    edges = np.histogram_bin_edges(forecasts, bins=bins, range=(low_bin, high_bin))
    binned_forecasts = digitize_minus_one(forecasts, edges)
    binned_observations = digitize_minus_one(observations, edges)


    # Append one observation in each bin-category to the forecasts to prevent 0 probability occuring.
    unique_categories = np.arange(0, len(bins))
    binned_forecasts = np.concatenate((binned_forecasts, np.tile(unique_categories, binned_forecasts.shape[:-1] + (1,))), axis = -1)
    
    n = binned_forecasts.shape[-1]

    #if observations.shape == forecasts.shape:
        # exact prediction yields 0 ign
    ign_score = np.empty_like(binned_observations, dtype = float)
    for index in np.ndindex(ign_score.shape):
        ign_score[index] = _ensemble_ignorance_score(binned_forecasts[index], n, binned_observations[index])
    
    
    return ign_score