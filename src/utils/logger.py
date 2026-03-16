"""
Standardized JSON Logger
Production-grade logging for AI application monitoring.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "name": record.name
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

class StructuredLogger:
    def __init__(self, name: str, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers if already initialized
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

if __name__ == "__main__":
    logger = StructuredLogger("test-logger")
    logger.info("Initializing Aether Suite...", component="loader")
    logger.error("Sample Error Log", error_code=500)
