import logging

import src.elements.s3_parameters as s3p
import src.elements.text_attributes as txa
import src.functions.streams


class Foci:

    def __init__(self, s3_parameters: s3p.S3Parameters):

        self.__s3_parameters = s3_parameters
        self.__streams = src.functions.streams.Streams()

    def exc(self):

        uri = f's3://{self.__s3_parameters.internal}/warning/data.csv'
        text = txa.TextAttributes(uri=uri, header=0)
        frame = self.__streams.read(text=text)

        logging.info(frame.tail())
