# app/core/logger.py
import logging
_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def setup_logging(level: str = "DEBUG"):  # ✅ DEBUG로 올림
    logging.basicConfig(level=level, format=_fmt)
