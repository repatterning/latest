"""Module master.py"""
import typing

import pandas as pd


class Master(typing.NamedTuple):
    """
    The data type class ⇾ Master<br><br>

    Attributes<br>
    ----------<br>
    <b>training</b> : pandas.DataFrame
        The training data of a gauge<br>

    <b>testing</b> : pandas.DataFrame
        The testing data of a gauge<br>

    """

    training: pd.DataFrame
    testing: pd.DataFrame
