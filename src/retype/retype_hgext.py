import io
import token
import tokenize
from typing import cast

try:
    from mercurial import replacetokens
except ImportError:
    # NOTE: the following is taken directly from `mercurial/__init__.py` in
    # Mercurial 4.1.2.
    # Code by Gregory Szorc, Martijn Pieters, Yuya Nishihara, and Pulkit Goyal.

    def replacetokens(tokens, fullname):
        """Transform a stream of tokens from raw to Python 3.

        It is called by the custom module loading machinery to rewrite
        source/tokens between source decoding and compilation.

        Returns a generator of possibly rewritten tokens.

        The input token list may be mutated as part of processing. However,
        its changes do not necessarily match the output token stream.
        """
        futureimpline = False

        # The following utility functions access the tokens list and i index of
        # the for i, t enumerate(tokens) loop below
        def _isop(j, *o):
            """Assert that tokens[j] is an OP with one of the given values"""
            try:
                return tokens[j].type == token.OP and tokens[j].string in o
            except IndexError:
                return False

        def _findargnofcall(n):
            """Find arg n of a call expression (start at 0)

            Returns index of the first token of that argument, or None if
            there is not that many arguments.

            Assumes that token[i + 1] is '('.

            """
            nested = 0
            for j in range(i + 2, len(tokens)):
                if _isop(j, ")", "]", "}"):
                    # end of call, tuple, subscription or dict / set
                    nested -= 1
                    if nested < 0:
                        return None
                elif n == 0:
                    # this is the starting position of arg
                    return j
                elif _isop(j, "(", "[", "{"):
                    nested += 1
                elif _isop(j, ",") and nested == 0:
                    n -= 1

            return None

        def _ensureunicode(j):
            """Make sure the token at j is a unicode string

            This rewrites a string token to include the unicode literal prefix
            so the string transformer won't add the byte prefix.

            Ignores tokens that are not strings. Assumes bounds checking has
            already been done.

            """
            st = tokens[j]
            if st.type == token.STRING and st.string.startswith(("'", '"')):
                tokens[j] = st._replace(string="u%s" % st.string)

        for i, t in enumerate(tokens):
            # Convert most string literals to byte literals. String literals
            # in Python 2 are bytes. String literals in Python 3 are unicode.
            # Most strings in Mercurial are bytes and unicode strings are rare.
            # Rather than rewrite all string literals to use ``b''`` to indicate
            # byte strings, we apply this token transformer to insert the ``b``
            # prefix nearly everywhere.
            if t.type == token.STRING:
                s = t.string

                # Preserve docstrings as string literals. This is inconsistent
                # with regular unprefixed strings. However, the
                # "from __future__" parsing (which allows a module docstring to
                # exist before it) doesn't properly handle the docstring if it
                # is b''' prefixed, leading to a SyntaxError. We leave all
                # docstrings as unprefixed to avoid this. This means Mercurial
                # components touching docstrings need to handle unicode,
                # unfortunately.
                if s[0:3] in ("'''", '"""'):
                    yield t
                    continue

                # If the first character isn't a quote, it is likely a string
                # prefixing character (such as 'b', 'u', or 'r'. Ignore.
                if s[0] not in ("'", '"'):
                    yield t
                    continue

                # String literal. Prefix to make a b'' string.
                yield t._replace(string="b%s" % t.string)
                continue

            # Insert compatibility imports at "from __future__ import" line.
            # No '\n' should be added to preserve line numbers.
            if (
                t.type == token.NAME
                and t.string == "import"
                and all(u.type == token.NAME for u in tokens[i - 2 : i])
                and [u.string for u in tokens[i - 2 : i]] == ["from", "__future__"]
            ):
                futureimpline = True
            if t.type == token.NEWLINE and futureimpline:
                futureimpline = False
                if fullname == "mercurial.pycompat":
                    yield t
                    continue
                r, c = t.start
                l = (
                    b"; from mercurial.pycompat import "
                    b"delattr, getattr, hasattr, setattr, xrange, open\n"
                )
                for u in tokenize.tokenize(io.BytesIO(l).readline):
                    if u.type in (tokenize.ENCODING, token.ENDMARKER):
                        continue
                    yield u._replace(start=(r, c + u.start[1]), end=(r, c + u.end[1]))
                continue

            # This looks like a function call.
            if t.type == token.NAME and _isop(i + 1, "("):
                fn = t.string

                # *attr() builtins don't accept byte strings to 2nd argument.
                if fn in ("getattr", "setattr", "hasattr", "safehasattr") and not _isop(
                    i - 1, "."
                ):
                    arg1idx = _findargnofcall(1)
                    if arg1idx is not None:
                        _ensureunicode(arg1idx)

                # .encode() and .decode() on str/bytes/unicode don't accept
                # byte strings on Python 3.
                elif fn in ("encode", "decode") and _isop(i - 1, "."):
                    for argn in range(2):
                        argidx = _findargnofcall(argn)
                        if argidx is not None:
                            _ensureunicode(argidx)

                # It changes iteritems to items as iteritems is not
                # present in Python 3 world.
                elif fn == "iteritems":
                    yield t._replace(string="items")
                    continue

            # Emit unmodified token.
            yield t


def apply_job_security(code):
    """Treat input `code` like Python 2 (implicit strings are byte literals).

    The implementation is horribly inefficient but the goal is to be compatible
    with what Mercurial does at runtime.
    """
    buf = io.BytesIO(code.encode("utf8"))
    tokens = tokenize.tokenize(buf.readline)
    # NOTE: by setting the fullname to `mercurial.pycompat` below, we're
    # ensuring that hg-specific pycompat imports aren't inserted to the code.
    data = tokenize.untokenize(replacetokens(list(tokens), "mercurial.pycompat"))
    return cast(str, data.decode("utf8"))
