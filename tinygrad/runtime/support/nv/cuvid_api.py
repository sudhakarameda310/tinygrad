import ctypes
from ctypes import POINTER, c_int, c_uint, c_void_p, c_ulonglong

# CUVID Types
CUvideodecoder = c_void_p
CUvideoparser = c_void_p
CUdeviceptr = c_ulonglong

# CUVID Functions
cuvidCreateVideoParser = ctypes.CFUNCTYPE(c_int, POINTER(CUvideoparser), c_void_p)
cuvidDestroyVideoParser = ctypes.CFUNCTYPE(c_int, CUvideoparser)
cuvidCreateDecoder = ctypes.CFUNCTYPE(c_int, POINTER(CUvideodecoder), c_void_p)
cuvidDestroyDecoder = ctypes.CFUNCTYPE(c_int, CUvideodecoder)
cuvidDecodePicture = ctypes.CFUNCTYPE(c_int, CUvideodecoder, c_void_p)
cuvidParseVideoData = ctypes.CFUNCTYPE(c_int, CUvideoparser, c_void_p, c_uint)
cuvidMapVideoFrame = ctypes.CFUNCTYPE(c_int, CUvideodecoder, c_int, POINTER(c_void_p), POINTER(c_uint))
cuvidUnmapVideoFrame = ctypes.CFUNCTYPE(c_int, CUvideodecoder, CUdeviceptr)

# Load CUVID library
_libcuvid = None
try:
    _libcuvid = ctypes.CDLL("libnvcuvid.so.1")
    cuvidCreateVideoParser = _libcuvid.cuvidCreateVideoParser
    cuvidDestroyVideoParser = _libcuvid.cuvidDestroyVideoParser
    cuvidCreateDecoder = _libcuvid.cuvidCreateDecoder
    cuvidDestroyDecoder = _libcuvid.cuvidDestroyDecoder
    cuvidDecodePicture = _libcuvid.cuvidDecodePicture
    cuvidParseVideoData = _libcuvid.cuvidParseVideoData
    cuvidMapVideoFrame = _libcuvid.cuvidMapVideoFrame
    cuvidUnmapVideoFrame = _libcuvid.cuvidUnmapVideoFrame
except OSError:
    pass  # CUVID library not available
