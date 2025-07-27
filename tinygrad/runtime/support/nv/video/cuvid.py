from __future__ import annotations
import ctypes
from typing import Optional
from tinygrad.helpers import init_c_var, init_c_struct_t
from tinygrad.runtime.ops_cuda import check, CUDADevice
from .cuvid_core import (
    _libcuvid, cudaVideoCodec,
    CUvideodecoder, CUvideoparser, CUdeviceptr,
    cuvidCreateVideoParser, cuvidDestroyVideoParser,
    cuvidCreateDecoder, cuvidDestroyDecoder,
    cuvidDecodePicture, cuvidParseVideoData,
    cuvidMapVideoFrame, cuvidUnmapVideoFrame
)

class CUVIDDecoder:
    def __init__(self, dev: CUDADevice, codec: cudaVideoCodec = cudaVideoCodec.HEVC,
                 max_width: int = 3840, max_height: int = 2160):
        """Initialize NVIDIA video decoder for hardware-accelerated decoding

        Args:
            dev: CUDA device to use for decoding
            codec: Video codec to decode (default: HEVC)
            max_width: Maximum video width to support (default: 3840/4K)
            max_height: Maximum video height to support (default: 2160/4K)
        """
        if _libcuvid is None:
            raise RuntimeError("CUVID library not available")

        self.dev = dev
        self.codec = codec
        check(self.dev.context)

        # Create video parser
        parser_params = init_c_struct_t([
            ("CodecType", ctypes.c_uint32),
            ("ulMaxNumDecodeSurfaces", ctypes.c_uint32),
            ("ulClockRate", ctypes.c_uint32),
            ("ulErrorThreshold", ctypes.c_uint32),
            ("ulMaxDisplayDelay", ctypes.c_uint32),
            ("uReserved1", ctypes.c_uint32 * 5),
            ("pUserData", ctypes.c_void_p),
            ("pfnSequenceCallback", ctypes.c_void_p),
            ("pfnDecodePicture", ctypes.c_void_p),
            ("pfnDisplayPicture", ctypes.c_void_p),
        ])(
            CodecType=codec,
            ulMaxNumDecodeSurfaces=1,
            ulClockRate=0,
            ulErrorThreshold=100,
            ulMaxDisplayDelay=0
        )
        
        self.parser = init_c_var(CUvideoparser(),
                               lambda x: cuvidCreateVideoParser(ctypes.byref(x),
                                                             ctypes.byref(parser_params)))

        # Create decoder
        create_info = init_c_struct_t([
            ("ulWidth", ctypes.c_uint),
            ("ulHeight", ctypes.c_uint),
            ("ulNumDecodeSurfaces", ctypes.c_uint),
            ("CodecType", ctypes.c_uint),
            ("ChromaFormat", ctypes.c_uint),
            ("ulCreationFlags", ctypes.c_uint),
            ("bitDepthMinus8", ctypes.c_uint),
            ("ulIntraDecodeOnly", ctypes.c_uint),
            ("ulMaxWidth", ctypes.c_uint),
            ("ulMaxHeight", ctypes.c_uint),
        ])(
            ulWidth=max_width,
            ulHeight=max_height,
            ulNumDecodeSurfaces=1,
            CodecType=codec,
            ChromaFormat=1,  # NV12
            ulCreationFlags=0,
            bitDepthMinus8=0,
            ulIntraDecodeOnly=0,
            ulMaxWidth=max_width,
            ulMaxHeight=max_height
        )
        self.decoder = init_c_var(CUvideodecoder(),
                                lambda x: cuvidCreateDecoder(ctypes.byref(x),
                                                          ctypes.byref(create_info)))

    def decode_frame(self, bitstream: bytes, timestamp: int = 0) -> Optional[CUdeviceptr]:
        """Decode a single compressed video frame

        Args:
            bitstream: Compressed video frame data
            timestamp: Frame timestamp (optional)

        Returns:
            CUDA device pointer to decoded frame, or None if decoding failed
        """
        # Create bitstream buffer
        bitstream_len = len(bitstream)
        pic_params = init_c_struct_t([
            ("PicWidthInMbs", ctypes.c_uint),
            ("FrameHeightInMbs", ctypes.c_uint),
            ("CurrPicIdx", ctypes.c_int),
            ("field_pic_flag", ctypes.c_uint),
            ("progressive_frame", ctypes.c_uint),
            ("timestamp", ctypes.c_ulonglong),
            ("ref_pic_flag", ctypes.c_uint),
        ])(timestamp=timestamp)

        # Parse bitstream
        bitstream_data = (ctypes.c_uint8 * bitstream_len)(*bitstream)
        check(cuvidParseVideoData(self.parser, ctypes.byref(bitstream_data), bitstream_len))

        # Decode frame
        check(cuvidDecodePicture(self.decoder, ctypes.byref(pic_params)))

        # Map frame
        map_info = init_c_struct_t([
            ("mem_type", ctypes.c_uint),
            ("device_ptr", CUdeviceptr),
            ("pitch", ctypes.c_uint),
        ])()
        
        if cuvidMapVideoFrame(self.decoder, pic_params.CurrPicIdx,
                            ctypes.byref(map_info), None) == 0:
            return map_info.device_ptr
        return None

    def __del__(self):
        """Clean up decoder resources"""
        if hasattr(self, 'decoder'):
            cuvidDestroyDecoder(self.decoder)
        if hasattr(self, 'parser'):
            cuvidDestroyVideoParser(self.parser)
