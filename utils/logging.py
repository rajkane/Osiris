from loguru import logger


class LogManager:
    """Manages application-wide logging configuration and loggers using loguru."""

    _configured = False

    @classmethod
    def get_logger(cls):
        if not cls._configured:
            # Configure loguru logger only once
            logger.remove()
            logger.add(
                sink=lambda msg: print(msg, end=""),
                format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
                level="INFO",
            )
            cls._configured = True
        return logger

    @classmethod
    def set_level(cls, level):
        # Set loguru logger level
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="[{time:YYYY-MM-DD HH:mm:ss}] {level}: {message}",
            level=level,
        )
        cls._configured = True
