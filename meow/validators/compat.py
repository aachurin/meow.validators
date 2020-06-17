import sys


__all__ = ("get_args", "get_origin")


if sys.version_info >= (3, 8, 3):  # pragma: nocover
    from typing import get_origin, get_args
else:  # pragma: nocover
    import typing
    from typing import _GenericAlias, Generic  # type: ignore
    import collections.abc

    def get_origin(tp: typing.Any) -> typing.Optional[typing.Any]:
        if isinstance(tp, _GenericAlias):
            return tp.__origin__
        if tp is Generic:
            return Generic
        return None

    def get_args(tp: typing.Any) -> typing.Tuple[typing.Any, ...]:
        if isinstance(tp, _GenericAlias) and not tp._special:
            res = tp.__args__
            if get_origin(tp) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
            return res  # type: ignore
        return ()
