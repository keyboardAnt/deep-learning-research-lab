from types import ModuleType
import logging
from functools import partial, partialmethod


def get_logging() -> ModuleType:
    logging.INFO_IR = 25
    logging.addLevelName(logging.INFO_IR, 'INFO_IR')
    logging.Logger.info_ir = partialmethod(logging.Logger.log, logging.INFO_IR)
    logging.info_ir = partial(logging.log, logging.INFO_IR)
    return logging
