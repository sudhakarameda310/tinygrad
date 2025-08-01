#!/usr/bin/env python
import unittest
import sys
from tinygrad import Device
from tinygrad.runtime.support.nv.video.cuvid_core import _libcuvid, cudaVideoCodec
from tinygrad.runtime.support.nv.video.cuvid import CUVIDDecoder

CUVID_AVAILABLE = sys.platform == "linux" and _libcuvid is not None

@unittest.skipUnless(CUVID_AVAILABLE, "CUVID library not available")
class TestCUVIDDecoder(unittest.TestCase):
  def setUp(self):
    self.dev = Device["cuda"]
    self.decoder = CUVIDDecoder(self.dev, codec=cudaVideoCodec.CUDA_VIDEO_CODEC_HEVC)

  def tearDown(self):
    del self.decoder

  def test_hevc_decode(self):
    # Test with a small HEVC bitstream
    # This is just a minimal HEVC NAL unit for testing
    test_hevc = bytes.fromhex(
      "0000000140010C01FFFF01600000003000B0000003000003"
      "00099400000300000300000300000300000300062EE1F2"
    )
    
    # Decode frame
    decoded_ptr = self.decoder.decode_frame(test_hevc)
    self.assertIsNotNone(decoded_ptr, "Decoding failed")

if __name__ == '__main__':
  unittest.main()
