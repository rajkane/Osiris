import pytest

from utils.error import ErrorManager


def test_handle_error_print():
    try:
        ErrorManager.handle_error(Exception("test error"))
    except Exception:
        pytest.fail("handle_error raised Exception unexpectedly!")


def test_raise_if():
    with pytest.raises(ValueError):
        ErrorManager.raise_if(True, ValueError("error!"))
    # Should not raise
    ErrorManager.raise_if(False, ValueError("error!"))
