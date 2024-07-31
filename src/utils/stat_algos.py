def ewmv(x: float, mean: float, var: float, alpha: float = 0.05) -> tuple[float, float]:
    """
    Calculates the exponentially weighted moving variance for each new data point

    Args:
        x: new data point
        mean: EWMA of the previous points
        var: EWMV of the previous points
        alpha: weight of the new data point

    Returns:
        updated EWMA
        updated EWMV
    """
    diff = x - mean
    incr = alpha * diff
    new_mean = mean + incr
    new_var = (1 - alpha) * (diff * incr + var)
    return new_mean, new_var