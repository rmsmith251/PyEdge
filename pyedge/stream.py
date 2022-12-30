import logging
import time
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, PositiveInt

from pyedge.base import BaseProcessor

logger = logging.getLogger(__name__)


class Resize(BaseModel):
    height: PositiveInt = 500
    width: PositiveInt = 500


class Stream(BaseModel, arbitrary_types_allowed=True, extra='allow'):
    name: str
    url: str
    resize: Optional[Resize]
    _video: Optional[cv2.VideoCapture] = None

    def initialize_video(self) -> None:
        self._video = cv2.VideoCapture(self.url)
        status, _ = self._video.read()
        if not status:
            raise ConnectionError(f"Unable to read frame from {self.url}")

    def __call__(self) -> np.ndarray:
        if self._video is None:
            self.initialize_video()
        status, frame = self._video.read()
        if not status:
            logger.error(f"Error reading from stream {self.url}")
            return np.array([])

        if self.resize is not None:
            frame = cv2.resize(frame, (self.resize.height, self.resize.width))

        return frame

    def close(self) -> None:
        self._video.release()


class StreamConfig(BaseModel):
    streams: List[Stream]


class StreamProcessor(BaseProcessor):
    def __init__(self, config: StreamConfig, timeout: float = 0.001):
        super().__init__()
        self.config = config
        self.streams = config.streams
        self.timeout = timeout

    def release(self) -> None:
        for stream in self.streams:
            stream.close()

    def get_frames(self) -> List[np.ndarray]:
        frames = []
        for stream in self.streams:
            frames.append(stream())

    def run_forever(self) -> None:

        while not self.stop_event.is_set():
            start = time.time()
            frames = self.get_frames()
            self.out_q.put(frames, self.timeout)
            end = time.time()
            self._track_performance(len(self.streams) / (end - start))

        self._log_performance()
