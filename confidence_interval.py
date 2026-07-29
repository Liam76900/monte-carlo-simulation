import numpy as np
from scipy.stats import norm

def confidence_interval(values, alpha=0.05):
    num = len(values)
    sd = np.std(values,ddof=1)
    mean = np.mean(values)
    z = norm.ppf(1-(alpha/2))
    confidence_interval_upper = mean + (z * sd/np.sqrt(num))
    confidence_interval_lower = mean - (z * sd/np.sqrt(num))
    return (confidence_interval_lower, confidence_interval_upper)