# meow.validators
#
# Copyright (c) 2020-present Andrey Churin (aachurin@gmail.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import datetime
import uuid
import typing
import enum
import collections.abc
from dataclasses import field as _field, fields, is_dataclass, MISSING
from .compat import get_origin, get_args
from .elements import (
    Any,
    Validator,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Time,
    Date,
    UUID,
    Enum,
    TypedObject,
    Mapping,
    TypedMapping,
    List,
    TypedList,
    Tuple,
    TypedTuple,
    Set,
    FrozenSet,
    Optional,
    Union,
)


_T = typing.TypeVar("_T")
_T_co = typing.TypeVar("_T_co", covariant=True)


def field(  # type: ignore
    *,
    default=MISSING,
    default_factory=MISSING,
    repr=True,
    hash=None,
    init=True,
    compare=True,
    **kwargs,
):
    return _field(  # type: ignore
        default=default,
        default_factory=default_factory,
        repr=repr,
        hash=hash,
        init=init,
        compare=compare,
        metadata=kwargs,
    )


_Factory = typing.Callable[..., Validator[typing.Any]]
_Callback = typing.Callable[
    [typing.Type[object], typing.Tuple[Validator[typing.Any], ...]], _Factory,
]


class Container:
    _builtins: typing.Dict[typing.Type[typing.Any], _Factory] = {
        str: String,
        float: Float,
        int: Integer,
        bool: Boolean,
        datetime.datetime: DateTime,
        datetime.time: Time,
        datetime.date: Date,
        uuid.UUID: UUID,
        set: (lambda items=Any, **spec: Set(items, **spec)),
        frozenset: (lambda items=Any, **spec: FrozenSet(items, **spec)),
        list: (lambda items=Any, **spec: List(items, **spec)),
        dict: (lambda keys=Any, values=Any, **spec: Mapping(keys, values, **spec)),
        tuple: (lambda items=Any, **spec: Tuple(items, **spec)),
        collections.abc.Sequence: (lambda items=Any, **spec: Tuple(items, **spec)),
        collections.abc.MutableSequence: (
            lambda items=Any, **spec: Tuple(items, **spec)
        ),
        collections.abc.Set: (lambda items=Any, **spec: FrozenSet(items, **spec)),
        collections.abc.MutableSet: (lambda items=Any, **spec: Set(items, **spec)),
        collections.abc.Mapping: (
            lambda keys=Any, values=Any, **spec: Mapping(keys, values, **spec)
        ),
    }

    _lookup_cache: dict  # type: ignore

    def __init__(
        self, lookup_cache_size: int = 5000, default: typing.Optional[_Callback] = None,
    ):
        self._lookup_cache = {}
        self._lookup_cache_size = lookup_cache_size
        self._default = default

    # @classmethod
    # def is_primitive_type(cls, tp: typing.Type[object]) -> bool:
    #     return tp in cls._builtin_primitives
    #
    # @staticmethod
    # def is_enum_type(tp: typing.Type[object]) -> bool:
    #     return isinstance(tp, type) and issubclass(tp, enum.Enum)
    #
    # @staticmethod
    # def is_dataclass_type(tp: typing.Type[object]) -> bool:
    #     return isinstance(tp, type) and is_dataclass(tp)

    @typing.overload
    def get_validator(self, tp: typing.Type[_T]) -> Validator[_T]:
        ...  # pragma: nocover

    @typing.overload
    def get_validator(self, tp: _T) -> Validator[_T]:
        ...  # pragma: nocover

    def get_validator(self, tp):  # type: ignore
        try:
            return self._lookup_cache[tp]
        except KeyError:
            pass
        validator = self._lookup_cache[tp] = self._get_validator(tp, {})
        if len(self._lookup_cache) > self._lookup_cache_size:  # pragma: nocover
            self._lookup_cache.pop(next(iter(self._lookup_cache)))
        return validator

    __getitem__ = get_validator

    @typing.overload
    def get_validator_spec(self, tp: typing.Type[_T], **spec: object) -> Validator[_T]:
        ...  # pragma: nocover

    @typing.overload
    def get_validator_spec(self, tp: _T, **spec: object) -> Validator[_T]:
        ...  # pragma: nocover

    def get_validator_spec(self, tp, **spec):  # type: ignore
        return self._get_validator(tp, spec)

    __call__ = get_validator_spec

    @typing.overload
    def _get_validator(
        self, tp: typing.Type[_T], spec: typing.Dict[str, object]
    ) -> Validator[_T]:
        ...  # pragma: nocover

    @typing.overload
    def _get_validator(self, tp: _T, spec: typing.Dict[str, object]) -> Validator[_T]:
        ...  # pragma: nocover

    def _get_validator(self, tp, spec):  # type: ignore
        if tp is typing.Any:
            return Any

        if isinstance(tp, type):
            if factory := self._builtins.get(tp):
                return factory(**spec)

            if issubclass(tp, enum.Enum):
                assert not spec, "Spec for enums is not allowed"
                # noinspection PyTypeChecker
                return Enum(items=tp)

            if is_dataclass(tp):
                assert not spec, "Spec for dataclasses is not allowed"
                properties: typing.Dict[str, Validator[typing.Any]] = {}
                required = []
                for fld in fields(tp):
                    if not fld.init:
                        continue
                    if fld.default is MISSING and fld.default_factory is MISSING:  # type: ignore
                        required.append(fld.name)
                    if fld.metadata:
                        properties[fld.name] = self._get_validator(
                            fld.type, dict(fld.metadata)
                        )
                    else:
                        properties[fld.name] = self.get_validator(fld.type)
                return TypedObject(properties, tp, required=tuple(required))

            if self._default is not None and (resolved := self._default(tp, ())):
                # noinspection PyUnboundLocalVariable
                return resolved(**spec)

        elif origin := get_origin(tp):
            type_args = get_args(tp)
            items: typing.Any = spec.pop("items", None)

            if origin is typing.Union:
                none_type = type(None)
                if none_type in type_args:
                    args = tuple(item for item in type_args if item is not none_type)
                    inner_tp = (
                        args[0] if len(args) == 1 else typing.Union.__getitem__(args)
                    )
                    if spec:
                        validator = self._get_validator(inner_tp, spec)
                    else:
                        validator = self.get_validator(inner_tp)
                    return Optional(validator)
                if items is None:
                    items = tuple(self.get_validator(arg) for arg in type_args)
                else:
                    # simple check, if the developer specifies the inconsistent validators, it's his problem
                    assert isinstance(items, tuple) and len(items) == len(type_args)
                assert not spec, "Invalid spec for Union"
                return Union(items=items)

            if origin is tuple:
                if items is not None:
                    if type_args and type_args[-1] is not ...:
                        assert isinstance(items, tuple) and len(items) == len(type_args)
                        return TypedTuple(items, **spec)
                elif not type_args:
                    return Tuple(Any, **spec)
                elif type_args[-1] is ...:
                    return Tuple(self.get_validator(type_args[0]), **spec)
                else:
                    return TypedTuple(
                        tuple(self.get_validator(arg) for arg in type_args), **spec
                    )

            # handle other generics
            if items is None:
                items = [self.get_validator(type_arg) for type_arg in type_args]
            if factory := self._builtins.get(origin):
                return factory(*items, **spec)

            if self._default is not None and (resolved := self._default(tp, items)):
                return resolved(**spec)

        raise TypeError("Don't know how to create validator for %r" % tp)


V = Container()
get_validator = V.get_validator
