# vision_server/camera_client.py
import threading
import struct
from typing import Optional, Tuple

import numpy as np
import zmq


FRAME_HEADER_FMT = "<IIIIQQ"  # width,height,channels,bpp,pixfmt,timestamp


class ZmqFrameSubscriber(threading.Thread):
    def __init__(self, url: str, name: str = "cam"):
        super().__init__(daemon=True)
        self.url = url
        self.name = name
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._header = None

    def run(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self.url)
        sock.setsockopt(zmq.SUBSCRIBE, b"")

        header_size = struct.calcsize(FRAME_HEADER_FMT)

        while True:
            msg = sock.recv()
            header_bytes = msg[:header_size]
            width, height, channels, bpp, pixfmt, ts = struct.unpack(
                FRAME_HEADER_FMT, header_bytes
            )
            data = msg[header_size:]

            if len(data) != width * height * channels * bpp:
                # tamaño inesperado, lo ignoramos
                continue

            if bpp == 1:
                dtype = np.uint8
            elif bpp == 2:
                dtype = np.uint16
            else:
                dtype = np.uint8  # fallback

            arr = np.frombuffer(data, dtype=dtype).copy()
            if channels > 1:
                arr = arr.reshape((height, width, channels))
            else:
                arr = arr.reshape((height, width))

            with self._lock:
                self._frame = arr
                self._header = {
                    "w": width,
                    "h": height,
                    "channels": channels,
                    "bpp": bpp,
                    "pixfmt": pixfmt,
                    "timestamp": ts,
                }

    def latest(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), dict(self._header)
