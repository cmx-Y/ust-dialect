
import warnings
from contextvars import ContextVar

# By default, Python ignores deprecation warnings.
# we have to enable it to see the warning.
warnings.simplefilter("always", DeprecationWarning)

PrintLog = ContextVar("PrintLog", default=False)

class bcolors:
    """ANSI color escape codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class USTException(Exception):
    """Base class for all UST exceptions.

    Exception is the base class for warnings and errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.
    """

    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message


class USTWarning(USTException):
    """Base class for all UST warnings.

    Warning is the base class for all warnings.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The warning message.

    line : int, optional
        The line number of the warning.

    category_str : str, optional
        The warning category string.

    category : Warning, optional
        The warning category.
    """

    def __init__(self, message, line=None, category_str=None, category=None):
        message = bcolors.OKBLUE + message + bcolors.ENDC
        if category_str is not None:
            message = "\n{} {}".format(category_str, message)
        if line is not None:
            message += bcolors.BOLD + " (line {})".format(line) + bcolors.ENDC
        USTException.__init__(self, message)
        self.category = category

    def warn(self):
        warnings.warn(self.message, category=self.category)

    def log(self):
        if PrintLog.get():
            print(self.message)


class USTError(USTException):
    """Base class for all UST errors.

    Error is the base class for all errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.

    line: int, optional
        The line number of the error.

    category_str : str, optional
        The error category string.
    """

    def __init__(self, message, line=None, category_str=None):
        message = bcolors.OKBLUE + message + bcolors.ENDC
        if category_str is not None:
            message = "{} {}".format(category_str, message)
        if line is not None:
            message += bcolors.BOLD + " (line {})".format(line) + bcolors.ENDC
        USTException.__init__(self, message)

    def error(self):
        raise self.message

""" Inherited Error subclasses """

class DTypeError(USTError):
    """A subclass for specifying data type related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Data Type]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)


class APIError(USTError):
    """A subclass for specifying API related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[API]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)


class DSLError(USTError):
    """A subclass for specifying imperative DSL related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Imperative]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)


class TensorError(USTError):
    """A subclass for specifying tensor related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Tensor]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)


class DeviceError(USTError):
    """A subclass for specifying device related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Device]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)


class AssertError(USTError):
    """A subclass for specifying assert related exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Assert]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)

""" New Error subclasses """

class USTNotImplementedError(USTError):
    """A subclass for specifying not implemented exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Not Implemented]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)

class MLIRLimitationError(USTError):
    """A subclass for specifying MLIR limitation exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[MLIR Limitation]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)

class USTValueError(USTError):
    """A subclass for specifying UST value exception"""

    def __init__(self, msg, line=None):
        category_str = bcolors.FAIL + "[Value Error]" + bcolors.ENDC
        USTError.__init__(self, msg, line, category_str)

""" New Warning subclasses """
class DTypeWarning(USTWarning):
    """A subclass for specifying data type related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Data Type]" + bcolors.ENDC
        USTWarning.__init__(self, msg, line, category_str, RuntimeWarning)


class USTDeprecationWarning(USTWarning):
    """A subclass for specifying deprecation warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Deprecation]" + bcolors.ENDC
        USTWarning.__init__(self, msg, line, category_str, DeprecationWarning)

class APIWarning(USTWarning):
    """A subclass for specifying API related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[API]" + bcolors.ENDC
        USTWarning.__init__(self, msg, line, category_str, RuntimeWarning)

class PassWarning(USTWarning):
    """A subclass for specifying pass related warning"""

    def __init__(self, msg, line=None):
        category_str = bcolors.WARNING + "[Pass]" + bcolors.ENDC
        USTWarning.__init__(self, msg, line, category_str, RuntimeWarning)