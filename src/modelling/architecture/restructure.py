
import json

import pandas as pd


class Restructure:

    def __init__(self):
        pass

    @staticmethod
    def exc(data: pd.DataFrame) -> dict:

        string = data.to_json(orient='split')
        dictionary = json.loads(string)

        dictionary['index_names'] = None
        dictionary['column_names'] = None

        return dictionary
