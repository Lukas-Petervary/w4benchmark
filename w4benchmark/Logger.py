import logging
import sys

COLOR_CODES = {
    'DEBUG': '\033[94m',    # Blue
    'INFO': '\033[92m',     # Green
    'WARNING': '\033[93m',  # Yellow
    'ERROR': '\033[91m',    # Red
    'CRITICAL': '\033[95m', # Magenta
    'RESET': '\033[0m'
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLOR_CODES.get(record.levelname, COLOR_CODES['RESET'])
        reset = COLOR_CODES['RESET']
        message = super().format(record)
        return f"{color}{message}{reset}"

W4Logger = logging.getLogger("w4benchmark")
W4Logger.setLevel(logging.WARNING)

handler = logging.StreamHandler(sys.stdout)
formatter = ColoredFormatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
W4Logger.addHandler(handler)
