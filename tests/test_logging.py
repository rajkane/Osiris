from utils.logging import LogManager


def test_get_logger_returns_logger():
    logger = LogManager.get_logger()
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")


def test_set_level_does_not_crash():
    logger = LogManager.get_logger()
    LogManager.set_level("DEBUG")
    # We don't assert on stdout here because LogManager may route messages
    # through tqdm.write depending on environment.
    logger.debug("Debug message")
