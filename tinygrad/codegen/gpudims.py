import math
from tinygrad.uop.ops import UOp, Ops, sint, PatternMatcher, UPat, KernelInfo, ssimplify
from tinygrad.helpers import all_int
from tinygrad.dtype import dtypes
from tinygrad.shape.view import get_contraction
from tinygrad.renderer import Renderer

def _group_dims(dims:tuple[sint, ...], max_sizes:tuple[int, ...]):
  # TODO: symbolic shape
  if not all_int(dims): return dims
  while len(dims) > len(max_sizes) or any(d > m for d,m in zip(dims, max_sizes)):
    for i,m in enumerate(max_sizes):
      if i < (len(dims)-1) and dims[i] * dims[i+1] <= m:
        dims = dims[:i] + (dims[i]*dims[i+1],) + dims[i+2:]
        break
    else: return None
  return dims

def _split_dims(dims, max_sizes):
  if all(d <= m for d,m in zip(dims, max_sizes)): return dims
  _dims = list(dims) + [1]*(3-len(dims))
  for i in range(len(_dims)):
    while _dims[i] > max_sizes[i]:
      div = next((d for d in range(2, math.ceil(math.sqrt(_dims[i])) + 1) if (_dims[i] % d) == 0), 1)
      if div == 1: raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
      _dims[i], _dims[(i+1)%len(_dims)] = _dims[i]//div, _dims[(i+1)%len(_dims)]*div
  return tuple(_dims[:2] if _dims[2] == 1 else _dims[0] if _dims[1:3] == [1,1] else _dims)

def get_grouped_dims(prefix, dims:tuple[sint, ...], max_sizes:tuple[int, ...]|None, reverse=False) -> list[UOp]:
  if reverse: dims = dims[::-1]
  # try to group first: (a, b, c, d) -> (ab, c, d)
  limited = (grouped if (grouped := _group_dims(dims, max_sizes)) else dims) if max_sizes is not None else dims
  # check if grouping failed
  if max_sizes is not None and len(limited) > len(max_sizes): raise RuntimeError(f"cannot limit dim {dims=}, {max_sizes=}")
  # try to split up dims: (a,) -> (b, c)
  if limited == dims: limited = _split_dims(dims, max_sizes) if max_sizes is not None else dims
  ret = raw_idxs = [UOp(Ops.SPECIAL, dtypes.int, (), (f"{prefix}{i}", s)) for i,s in enumerate(limited)]
  if len(limited) < len(dims):
    ret = []
    if (contraction:=get_contraction(dims, limited)) is None: raise AssertionError(f"get_contraction should not be None {dims=} {limited=}")
    for idx, contraction_group in zip(raw_idxs, contraction):
      for c in contraction_group[:-1]:
        ret.append(idx % dims[c])
        idx //= dims[c]
      ret.append(idx)
  elif len(limited) > len(dims):
    a, b = len(limited), len(dims)
    if a == 2 and b == 1: ret = [raw_idxs[0] * limited[1] + raw_idxs[1]]
    if a == 3 and b == 1: ret = [raw_idxs[0] * (limited[1] * limited[2]) + raw_idxs[1] * limited[2] + raw_idxs[2]]
    if a == 3 and b == 2: ret = [raw_idxs[0] * limited[1] + raw_idxs[1], raw_idxs[2]]
  return ret[::-1] if reverse else ret

def add_gpudims(ctx:Renderer, s:UOp):
  if s.arg is None: return None
  ki: KernelInfo = s.arg
  if not ki.global_dims and not ki.local_dims: return None
  s_topo = list(s.toposort())
  if any(x.op is Ops.SPECIAL for x in s_topo): return None
  all_ranges = {x.arg:x for x in s_topo if x.op is Ops.RANGE}
  # NOTE: this supports globals/locals in any position
  ranges = [all_ranges[r] for r in ki.global_dims+ki.local_dims]
  global_shape = tuple([ssimplify(r.src[0]) for r in ranges if r.arg in ki.global_dims])
  local_shape = tuple([ssimplify(r.src[0]) for r in ranges if r.arg in ki.local_dims])
  if ki.dont_use_locals:
    assert not ki.local_dims, "can't use locals if there's no local dims"
    idxs = get_grouped_dims("idx", global_shape, ctx.global_max, reverse=True)
  else:
    # define indexes for GPU-like execution
    idxs = get_grouped_dims("gidx", global_shape, ctx.global_max, reverse=True) + get_grouped_dims("lidx", local_shape, ctx.local_max)
  return s.substitute(dict(zip(ranges, idxs)))

pm_add_gpudims = PatternMatcher([
  (UPat(Ops.SINK, name="s"), add_gpudims),
])
