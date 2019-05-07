from tokenize import TokenInfo
from typing import Iterator, List, Optional

# needed to bypass https://github.com/python/black/issues/837

def replacetokens(  # type: ignore
    tokens: List[TokenInfo], fullname: str
) -> Iterator[TokenInfo]:
    def _isop(j: int, *o: str) -> bool: ...
    def _findargnofcall(n: int) -> Optional[int]: ...
    def _ensureunicode(j: int) -> None: ...

def apply_job_security(code: str) -> str: ...
