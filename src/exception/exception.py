import sys
from src.logging.logger import logging

class NetworkSecurityException(Exception):
    """Base exception for network security errors."""
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_, exc_tb = error_details.exc_info()

        self.line_no = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occurred in script: {self.file_name} at line number: {self.line_no} with message: {self.error_message}"
    
if __name__ == "__main__":
    try:
        logging.info("This is a test log message.")
        a = 1 / 0
        print('This will not be printed', a)
    except Exception as e:
        raise NetworkSecurityException(e, sys)