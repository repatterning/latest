
import typing

import pandas as pd

class Master(typing.NamedTuple):
    """
    The data type class ⇾ Master<br><br>

    Attributes<br>
    ----------<br>
    <b>training</b> : pandas.DataFrame
        The training data of an institution<br>

    <b>testing</b> : pandas.DataFrame
        The testing data of an institution

    """

    training: pd.DataFrame
    testing: pd.DataFrame
