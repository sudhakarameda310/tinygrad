from __future__ import annotations
import sys
import ctypes
from enum import IntEnum
from ctypes import POINTER, c_int, c_uint, c_void_p, c_ulonglong

# Load CUVID library
_libcuvid = None
if sys.platform == "linux":  # CUVID is only supported on Linux
    try:
        _libcuvid = ctypes.CDLL("libnvcuvid.so.1")
    except OSError:
        pass

class cudaVideoCodec(IntEnum):
    """Video codec enums supported by NVDEC"""
    NONE = 0
    MPEG1 = 1
    MPEG2 = 2
    MPEG4 = 3
    VC1 = 4
    H264 = 5
    JPEG = 6
    H264_SVC = 7
    H264_MVC = 8
    HEVC = 9
    VP8 = 10
    VP9 = 11
    AV1 = 12

# CUVID Types
CUvideodecoder = c_void_p
CUvideoparser = c_void_p
CUdeviceptr = c_ulonglong

if _libcuvid is not None:
    # CUVID Functions
    cuvidCreateVideoParser = _libcuvid.cuvidCreateVideoParser
    cuvidCreateVideoParser.argtypes = [POINTER(CUvideoparser), c_void_p]
    cuvidCreateVideoParser.restype = c_int

    cuvidDestroyVideoParser = _libcuvid.cuvidDestroyVideoParser
    cuvidDestroyVideoParser.argtypes = [CUvideoparser]
    cuvidDestroyVideoParser.restype = c_int

    cuvidCreateDecoder = _libcuvid.cuvidCreateDecoder
    cuvidCreateDecoder.argtypes = [POINTER(CUvideodecoder), c_void_p]
    cuvidCreateDecoder.restype = c_int

    cuvidDestroyDecoder = _libcuvid.cuvidDestroyDecoder
    cuvidDestroyDecoder.argtypes = [CUvideodecoder]
    cuvidDestroyDecoder.restype = c_int

    cuvidDecodePicture = _libcuvid.cuvidDecodePicture
    cuvidDecodePicture.argtypes = [CUvideodecoder, c_void_p]
    cuvidDecodePicture.restype = c_int

    cuvidParseVideoData = _libcuvid.cuvidParseVideoData
    cuvidParseVideoData.argtypes = [CUvideoparser, c_void_p, c_uint]
    cuvidParseVideoData.restype = c_int

    cuvidMapVideoFrame = _libcuvid.cuvidMapVideoFrame
    cuvidMapVideoFrame.argtypes = [CUvideodecoder, c_int, POINTER(c_void_p), POINTER(c_uint)]
    cuvidMapVideoFrame.restype = c_int

    cuvidUnmapVideoFrame = _libcuvid.cuvidUnmapVideoFrame
    cuvidUnmapVideoFrame.argtypes = [CUvideodecoder, CUdeviceptr]
    cuvidUnmapVideoFrame.restype = c_int
