def check_subclass(obj, parent_class) -> bool:
    """
    This method checks whether an object instance is an instance of a subclass
    """
    return issubclass(obj.__class__, parent_class)

