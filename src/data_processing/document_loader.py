from pathlib import Path


def load_document(file_path):
    """
    Loads a text document.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    str
        Document contents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"{file_path} does not exist."
        )

    with open(path, "r", encoding="utf-8") as f:
        return f.read()