"""Module gauge.py"""
import typing


class Gauge(typing.NamedTuple):
    """
    The data type class ⇾ Gauge<br><br>

    Attributes<br>
    ----------<br>
    <b>ts_id</b> : int
        The identification code of a gauge's time series<br>

    <b>catchment_id</b> : int
        The catchment identification code of the gauge's catchment area.

    <b>gauge_datum</b> : float
        The reference point of a gauge site, in metres.
    """

    ts_id: int
    catchment_id: int
    gauge_datum: float
