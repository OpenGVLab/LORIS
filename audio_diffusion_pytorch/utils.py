from functools import reduce
from inspect import isfunction
from typing import Callable, List, Optional, Sequence, TypeVar, Union

from typing_extensions import TypeGuard

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def iff(condition: bool, value: T) -> Optional[T]:
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    return isinstance(obj, list) or isinstance(obj, tuple)


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)

def normalize(x):
    A = x.clone().squeeze()
    # print(f"A Shape: {A.shape}")
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0]
    x = A.unsqueeze(-1)
    # print(torch.max(x, dim=1, keepdim=False)[0])
    # print(torch.min(x, dim=1, keepdim=False)[0])
    return x