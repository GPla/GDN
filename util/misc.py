import typing as T
import collections


def flatten(
    mapping: T.Mapping[str, T.Any],
    *,
    prefix: str = '',
    sep: str = '.',
    flatten_list: bool = True
) -> T.Mapping[str, T.Any]:
    """Turns a nested mapping into a flattened mapping.

    Args:
        mapping (T.Mapping[str, T.Any]): Mapping to flatten.
        prefix (str): Prefix to preprend to the key.
        sep (str): Seperator of flattened keys.
        flatten_list (bool): Whether to flatten lists.

    Returns:
        T.Mapping[str, T.Any]: Returns a (flattened) mapping.

    Example:
        >>> flatten({
        ...    'flat1': 1,
        ...    'dict1': {'c': 1, 'd': 2},
        ...    'nested': {'e': {'c': 1, 'd': 2}, 'd': 2},
        ...    'list1': [1, 2],
        ...    'nested_list': [{'1': 1}]
        ... })
        {
            'flat1': 1,
            'dict1.c': 1,
            'dict1.d': 2,
            'nested.e.c': 1,
            'nested.e.d': 2,
            'nested.d': 2,
            'list1.0': 1,
            'list1.1': 2,
            'nested_list.0.1': 1
        }
    """

    items: T.List[T.Tuple[str, T.Any]] = []
    for k, v in mapping.items():
        key = f'{sep}'.join([prefix, k]) if prefix else k
        if isinstance(v, collections.Mapping):
            items.extend(flatten(
                v,
                prefix=key,
                sep=sep,
                flatten_list=flatten_list
            ).items())

        elif isinstance(v, list) and flatten_list:
            for i, v in enumerate(v):
                items.extend(flatten(
                    {str(i): v},
                    prefix=key,
                    sep=sep,
                    flatten_list=flatten_list
                ).items())

        else:
            items.append((key, v))

    return dict(items)


def clean_dict(mapping: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    """Cleans up a dict such that no custom python types are written.

       Args:
           mapping (T.Dict[str, T.Any]): Mapping to clean.

       Returns:
           T.Dict[str, T.Any]: Returns the cleaned mapping.
    """
    result = {}
    for k, v in flatten(mapping, sep='/').items():
        if hasattr(v, 'dtype'):  # numpy and torch custom types
            result[k] = v.item()
        else:
            result[k] = v

    return result
