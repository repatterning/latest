"""Module master.py"""
import typing

import pandas as pd


class Inference(typing.NamedTuple):
    """
    The data type class ⇾ Inference<br><br>

    Attributes<br>
    ----------<br>
    <b>evidence_lower_bound</b> : pandas.DataFrame
        Loss curve data<br>

    <b>estimates</b> : pandas.DataFrame
        Originals, etc<br>

    """

    evidence_lower_bound: pd.DataFrame
    estimates: pd.DataFrame
