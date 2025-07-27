from __future__ import annotations
import ctypes
from enum import IntEnum
from typing import Optional
from tinygrad.helpers import init_c_var, init_c_struct_t
from tinygrad.runtime.autogen.nv import nv
from tinygrad.runtime.ops_cuda import check, CUDADevice

class cudaVideoCodec(IntEnum):
  CUDA_VIDEO_CODEC_NONE = 0
  CUDA_VIDEO_CODEC_MPEG1 = 1
  CUDA_VIDEO_CODEC_MPEG2 = 2
  CUDA_VIDEO_CODEC_MPEG4 = 3
  CUDA_VIDEO_CODEC_VC1 = 4
  CUDA_VIDEO_CODEC_H264 = 5
  CUDA_VIDEO_CODEC_JPEG = 6
  CUDA_VIDEO_CODEC_H264_SVC = 7
  CUDA_VIDEO_CODEC_H264_MVC = 8
  CUDA_VIDEO_CODEC_HEVC = 9
  CUDA_VIDEO_CODEC_VP8 = 10
  CUDA_VIDEO_CODEC_VP9 = 11
  CUDA_VIDEO_CODEC_AV1 = 12

class CUVIDPARSERPARAMS(ctypes.Structure):
  _fields_ = [
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
    ("pfnGetOperatingPoint", ctypes.c_void_p),
    ("pfnGetSEIMsg", ctypes.c_void_p),
  ]

class CUVIDEOFORMAT(ctypes.Structure):
  _fields_ = [
    ("codec", ctypes.c_uint32),
    ("frame_rate", ctypes.c_struct_anon),
    ("progressive_sequence", ctypes.c_uint8),
    ("bit_depth_luma_minus8", ctypes.c_uint8),
    ("bit_depth_chroma_minus8", ctypes.c_uint8),
    ("min_num_decode_surfaces", ctypes.c_uint8),
    ("coded_width", ctypes.c_uint16),
    ("coded_height", ctypes.c_uint16),
    ("display_area", ctypes.c_struct_anon),
    ("chroma_format", ctypes.c_uint8),
    ("output_format", ctypes.c_uint8),
    ("reserved1", ctypes.c_uint8 * 6),
    ("coded_buf_size", ctypes.c_uint32),
    ("sei_payload_size", ctypes.c_uint32),
  ]

class CUVIDDecoder:
  def __init__(self, dev:CUDADevice, codec:cudaVideoCodec=cudaVideoCodec.CUDA_VIDEO_CODEC_HEVC, 
               max_width:int=3840, max_height:int=2160):
    self.dev = dev
    self.codec = codec
    check(nv.cuCtxSetCurrent(self.dev.context))
    
    # Create video parser
    parser_params = CUVIDPARSERPARAMS(
      CodecType=codec,
      ulMaxNumDecodeSurfaces=1,
      ulClockRate=0,
      ulErrorThreshold=100,
      ulMaxDisplayDelay=0
    )
    self.parser = init_c_var(nv.CUvideoparser(), 
                          lambda x: nv.cuvidCreateVideoParser(ctypes.byref(x), ctypes.byref(parser_params)))

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
      ("display_area", ctypes.c_uint * 4),
      ("Reserved", ctypes.c_uint * 4),
      ("target_rect", ctypes.c_uint * 4),
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
    self.decoder = init_c_var(nv.CUvideodecoder(),
                           lambda x: nv.cuvidCreateDecoder(ctypes.byref(x), ctypes.byref(create_info)))

  def decode_frame(self, bitstream:bytes, timestamp:int=0) -> Optional[nv.CUdeviceptr]:
    # Create bitstream buffer
    bitstream_len = len(bitstream)
    pic_params = init_c_struct_t([
      ("PicWidthInMbs", ctypes.c_uint),
      ("FrameHeightInMbs", ctypes.c_uint),
      ("CurrPicIdx", ctypes.c_int),
      ("field_pic_flag", ctypes.c_uint),
      ("bottom_field_flag", ctypes.c_uint),
      ("second_field", ctypes.c_uint),
      ("progressive_frame", ctypes.c_uint),
      ("timestamp", ctypes.c_ulonglong),
      ("ref_pic_flag", ctypes.c_uint),
      ("intra_pic_flag", ctypes.c_uint),
      ("Reserved", ctypes.c_uint * 32),
    ])(timestamp=timestamp)

    # Parse bitstream
    bitstream_data = (ctypes.c_uint8 * bitstream_len)(*bitstream)
    check(nv.cuvidParseVideoData(self.parser, ctypes.byref(bitstream_data), bitstream_len))

    # Decode frame
    check(nv.cuvidDecodePicture(self.decoder, ctypes.byref(pic_params)))
    
    # Map frame
    map_info = init_c_struct_t([
      ("mem_type", ctypes.c_uint),
      ("device_ptr", nv.CUdeviceptr),
      ("pitch", ctypes.c_uint),
      ("width", ctypes.c_uint),
      ("height", ctypes.c_uint),
    ])()
    if nv.cuvidMapVideoFrame(self.decoder, pic_params.CurrPicIdx, ctypes.byref(map_info), None) == 0:
      return map_info.device_ptr
    return None

  def __del__(self):
    try:
      check(nv.cuvidDestroyDecoder(self.decoder))
      check(nv.cuvidDestroyVideoParser(self.parser))
    except AttributeError:
      pass
