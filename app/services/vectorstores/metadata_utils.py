# ---------------------------------------------------
# Metadata sanitization
# ---------------------------------------------------

def sanitize_metadata(
    metadata: dict
) -> dict:
    """
    Remove metadata values that cannot be stored
    inside ChromaDB.

    Chroma supports:
    - str
    - int
    - float
    - bool
    - list of primitive values of the same type
    """

    cleaned = {}

    for key, value in metadata.items():

        if isinstance(
            value,
            (str, int, float, bool)
        ):
            cleaned[key] = value

        elif (
            isinstance(value, list)
            and all(
                isinstance(
                    item,
                    (str, int, float, bool)
                )
                for item in value
            )
        ):
            cleaned[key] = value

    return cleaned
