from .memory import _SingletonMeta


class ErrorManager(metaclass=_SingletonMeta):
    """Handles application-wide error management and reporting as a singleton."""

    def _handle_error(self, error: Exception, logger=None):
        if logger:
            logger.error(f"Error occurred: {error}")
        else:
            print(f"Error occurred: {error}")

    def _raise_if(self, condition: bool, error: Exception):
        if condition:
            raise error

    # Backwards-compatible classmethod wrappers that delegate to the singleton
    @classmethod
    def handle_error(cls, error: Exception, logger=None):
        cls()._handle_error(error, logger=logger)

    @classmethod
    def raise_if(cls, condition: bool, error: Exception):
        cls()._raise_if(condition, error)
