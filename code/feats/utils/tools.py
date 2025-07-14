# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np


def shannon_entropy(array):
    return -sum(np.power(array, 2) * np.log(np.spacing(1) + np.power(array, 2)))
