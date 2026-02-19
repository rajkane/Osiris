import pytest
from utils.logging import LogManager

def test_get_logger_returns_logger():
    logger = LogManager.get_logger()
    assert logger is not None
    assert hasattr(logger, 'info')
    assert hasattr(logger, 'error')

def test_set_level(capsys):
    logger = LogManager.get_logger()
    LogManager.set_level('DEBUG')
    logger.debug("Debug message should appear if level is set to DEBUG.")
    captured = capsys.readouterr()
    assert "Debug message should appear if level is set to DEBUG." in captured.out
