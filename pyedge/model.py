import time
from copy import deepcopy
from queue import Empty
from threading import Event
from typing import Dict, List, Tuple

import numpy as np

from pyedge.base import BaseProcessor
from pyedge.models import Classification, ModelConfig, ObjectDetection
from pyedge.utils import Message


class ModelProcessor(BaseProcessor):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.count_frames = 0

        if self.config.classification is not None:
            self.model = Classification(self.config.classification)
        elif self.config.detection is not None:
            self.model = ObjectDetection(self.config.detection)

    def get_batch(self) -> Tuple[List[np.ndarray], List[Message]]:
        images = []
        messages = []
        while len(images) <= self.config.max_batch_size:
            try:
                message = self.in_q.get_nowait()
            except Empty:
                break

            images.extend(message.frames)
            messages.append(message)
        return images, messages

    def unbatch(
        self, messages: List[Message], predictions: List[Dict]
    ) -> List[Message]:
        cur = 0
        for message in messages:
            preds = predictions[cur : cur + len(message.frames)]
            message.predictions = deepcopy(preds)
            cur += len(message.frames)

        return messages

    def run_forever(self, input_start_event: Event = None):
        """
        Try to avoid adding unnecessary processing here. The bottleneck will
        almost always be the model so let other processes handle other work.
        """
        while not self.stop_event.is_set():
            start = time.time()
            images, messages = self.get_batch()
            preds = self.model(images)
            messages = self.unbatch(messages, preds)
            self.out_q.put(messages)
            end = time.time()
            fps = len(images) / (end - start)
            self._track_performance(fps)

        self._log_performance()
