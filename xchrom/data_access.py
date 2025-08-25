import os
from pathlib import Path

# get data directory
def get_data_dir():
    """
    return the path of data directory

    Returns
    -------
    Path
        Absolute path to the test data directory

    Examples
    --------
    >>> data_dir = xchrom.get_data_dir()
    >>> print(f"data directory: {data_dir}")
    """
    base_path = Path(__file__).parent
    data_dir = base_path / "data"
    
    # if running in development mode, the path may be different
    if not data_dir.exists():
        data_dir = base_path.parent / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"cannot find data directory: {data_dir}")
    
    return data_dir


# list all available files
def list_items():
    """
    list all available files and directories in data directory

    Returns
    -------
    list
        List of file names and directories

    Examples
    --------
    >>> items = xchrom.list_items()
    >>> print(f"available files and directories: {items}")

    """
    data_dir = get_data_dir()
    return os.listdir(data_dir)


# ## for example
# import xchrom as xc
# # list all available files and directories
# print("available files and directories:", xc.list_items())
# # get data directory path
# print("data directory:", xc.get_data_dir())