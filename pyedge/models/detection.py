from typing import List

import numpy as np
import torch
import torchvision.models.detection as detection

MODEL_KEY = {"faster_rcnn": detection.fasterrcnn_resnet50_fpn}
device = "cuda" if torch.cuda.is_available() else "cpu"


class ObjectDetection:
    def __init__(self, model_name: str = "faster_rcnn", threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.model = MODEL_KEY[self.model_name](pretrained=True)
        self.dtype = np.float32

        # Warm up
        for _ in range(5):
            _ = self([np.random.rand(300, 300, 3).astype(self.dtype)])

    async def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        images = torch.from_numpy(np.asrray(images, dtype=self.dtype)).to(device)
        return images.permute(0, -1, 1, 2)

    async def postprocess(self, predictions: torch.Tensor) -> np.ndarray:
        output = []
        for pred in predictions:
            pred = pred.detach().cpu()
            boxes = pred["boxes"]
            labels = pred["labels"]
            scores = pred["scores"]
            masks = pred.get("masks", [])
            thresh = scores > self.threshold
            output.append(
                {
                    "boxes": np.asrray(boxes[thresh]),
                    "labels": np.asrray(labels[thresh]),
                    "scores": np.asrray(scores[thresh]),
                    "masks": np.asrray(masks[thresh] if masks else []),
                }
            )
        return output

    async def __call__(self, images: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = self.model(await self.preprocess(images))
        return await self.postprocess(out)


if __name__ == "__main__":
    model = ObjectDetection()
    breakpoint()
