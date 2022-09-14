from typing import List

import numpy as np
import torch
import torchvision.models.detection as detection
from pyedge.models.utils import DetectionConfig

MODEL_KEY = {"faster_rcnn": detection.fasterrcnn_resnet50_fpn}
device = "cuda" if torch.cuda.is_available() else "cpu"


class ObjectDetection:
    def __init__(self, config: DetectionConfig = DetectionConfig()):
        self.config = config
        self.model = MODEL_KEY[self.config.model_name](pretrained=True)

        # Warm up
        self([np.random.rand(300, 300, 3).astype(np.float32) for _ in range(5)])

    def preprocess(self, images: List[np.ndarray]):
        images = torch.from_numpy(np.array(images)).to(device)
        return images.permute(0, -1, 1, 2)

    def postprocess(self, predictions: torch.Tensor) -> np.ndarray:
        output = []
        for pred in predictions:
            pred = pred.detach().cpu()
            boxes = pred["boxes"]
            labels = pred["labels"]
            scores = pred["scores"]
            masks = pred.get("masks", [])
            thresh = scores > self.config.threshold
            output.append(
                {
                    "boxes": np.array(boxes[thresh]),
                    "labels": np.array(labels[thresh]),
                    "scores": np.array(scores[thresh]),
                    "masks": np.array(masks[thresh] if masks else []),
                }
            )
        return output

    def __call__(self, images: torch.Tensor):
        with torch.no_grad():
            out = self.model(self.preprocess(images))
        return self.postprocess(out)


if __name__ == "__main__":
    model = ObjectDetection()
    breakpoint()
