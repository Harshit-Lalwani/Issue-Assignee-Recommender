import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]
