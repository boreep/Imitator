import importlib
from imitator.utils.file_utils import install_octo


try:
    octo = importlib.import_module("octo")
except ImportError:
    print("octo not installed, installing...")
    install_octo()
    try:
        octo = importlib.import_module("octo")
    except ImportError:
        raise ImportError("octo installation failed")
