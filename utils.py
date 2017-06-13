import pickle


def pickle_dump(objects, filepaths):
    """
    Dump object(s) to filepath(s).
    :param objects: Single object or iterable of objects.
    :param filepaths: Single str filepath or iterable of filepath.
    """

    if isinstance(filepaths, str):
        filepaths = [filepaths]
        objects = [objects]

    assert len(objects) == len(filepaths)

    for obj, fp in zip(objects, filepaths):
        with open(fp, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(*filepaths):
    """
    Unpickle arbitrary number of filepaths.
    :param filepaths: Iterable of filepaths.
    :return: list of unpickled objects.
    """

    res = []
    for fp in filepaths:
        assert isinstance(fp, str)
        with open(fp, 'rb') as f:
            res.append(pickle.load(f))

    if len(filepaths) == 1:
        return res[0]

    return res

