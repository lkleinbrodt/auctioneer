from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
import logging

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent


def create_logger(name: str, level: str = 'INFO', file: Optional[str] = None) -> logging.Logger:
    ### Co-Pilots version
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if file:
            log_dir = Path(__file__).parent.parent / 'data' / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / file, 'w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    ### My version:
    # logger = logging.getLogger(name)
    # logger.propagate = False
    # logger.setLevel(level)
    # if sum([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]) == 0:
    #     ch = logging.StreamHandler()
    #     ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    #     logger.addHandler(ch)
    # if file is not None:
    #     if sum([isinstance(handler, logging.FileHandler) for handler in logger.handlers]) == 0:
    #         ch = logging.FileHandler(LOG_DIR/file, 'w')
    #         ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    #         logger.addHandler(ch)
            
    return logger

