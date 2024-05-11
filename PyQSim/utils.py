from functools import wraps

from .exceptions import ValidationError


def as_list(obj):
    """Return an object as a list."""
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def validate(validation_function, is_classmethod=False):
    if is_classmethod:

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    validation_function(*args, **kwargs)
                except Exception as e:
                    raise ValidationError(f"Validation failed: {str(e)}") from e
                return func(self, *args, **kwargs)

            return wrapper

        return decorator
    else:

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    validation_function(*args, **kwargs)
                except Exception as e:
                    raise ValidationError(f"Validation failed: {str(e)}") from e
                return func(*args, **kwargs)

            return wrapper

        return decorator
