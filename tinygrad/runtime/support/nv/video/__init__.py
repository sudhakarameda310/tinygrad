import sys

__all__ = ['cudaVideoCodec']

from .cuvid_core import cudaVideoCodec

if sys.platform == "linux":
    from .cuvid_core import _libcuvid
    from .cuvid import CUVIDDecoder
    __all__ += ['CUVIDDecoder', '_libcuvid']
