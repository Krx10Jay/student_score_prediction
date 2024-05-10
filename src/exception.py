import sys
from src import logger
import logging


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg= f'an Error occured in the python script name [{file_name}] at line number [{line_number}], error message: [{str(error)}]'

    return error_msg


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: str) -> None:
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_msg
    


    
        