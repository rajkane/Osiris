class ErrorManager:
    """Handles application-wide error management and reporting."""
    @staticmethod
    def handle_error(error: Exception, logger=None):
        if logger:
            logger.error(f"Error occurred: {error}")
        else:
            print(f"Error occurred: {error}")

    @staticmethod
    def raise_if(condition: bool, error: Exception):
        if condition:
            raise error

