# Generic utility functions for pre-process and batching symp. dataset

import numpy as np


def epiweek_to_month(ew):
    """
    Convert an epiweek to a month.
    """
    return (((ew - 40) + 52) % 52) // 4 + 1
