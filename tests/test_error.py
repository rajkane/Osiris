import unittest
from utils.error import ErrorManager

class TestErrorManager(unittest.TestCase):
    def test_handle_error_print(self):
        try:
            ErrorManager.handle_error(Exception("test error"))
        except Exception:
            self.fail("handle_error raised Exception unexpectedly!")

    def test_raise_if(self):
        with self.assertRaises(ValueError):
            ErrorManager.raise_if(True, ValueError("error!"))
        # Should not raise
        ErrorManager.raise_if(False, ValueError("error!"))

if __name__ == "__main__":
    unittest.main()

