import typing
import re
import sys

_output = None


def load_mypy_output():
    global _output
    if _output:
        return _output
    try:
        with open("mypy_tests.txt", "rt") as f:
            text = f.read()
    except IOError:
        raise IOError("Run mypy first")
    output: dict = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.search(r"^([^:]*):(\d+)", line)
        if match is not None:
            filename, num = match.groups()
            lineno = int(num)
            match = re.search(r"Revealed type is '([^']*)'", line)
            if match:
                (revealed_type,) = match.groups()
                output.setdefault(filename, {})[lineno] = revealed_type
    _output = output
    return output


if not typing.TYPE_CHECKING:

    def reveal_type(x):
        # for testing resolved types
        output = load_mypy_output()
        f = sys._getframe(1)
        filename = f.f_globals["__file__"]
        lineno = f.f_lineno
        ret = output.get(filename, {})
        if lineno in ret:
            return ret[lineno]
        lineno += 1
        if lineno in ret:
            return ret[lineno]
        lineno += 1
        return ret.get(lineno)
