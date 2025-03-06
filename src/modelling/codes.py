
import pandas as pd

import src.elements.codes as ce


class Codes:

    def __init__(self):
        pass

    @staticmethod
    def __structure(values: list[dict]) -> list[ce.Codes]:
        """

        :param values:
        :return:
        """

        return [ce.Codes(**value) for value in values]

    def exc(self, data: pd.DataFrame) -> list[ce.Codes]:
        """

        :param data:
        :return:
        """

        # Codes
        frame = data[['health_board_code', 'hospital_code']].drop_duplicates()
        frame.rename(columns={'health_board_code': 'board', 'hospital_code': 'institution'}, inplace=True)

        values: list[dict] = frame.reset_index(drop=True).to_dict(orient='records')

        return self.__structure(values=values)
