from tokenize import TokenInfo
from typing import Iterator, List

# needed to bypass https://github.com/python/black/issues/837

def replacetokens(tokens: List[TokenInfo], fullname: str) -> Iterator[TokenInfo]: ...
def apply_job_security(code: str) -> str: ...
