import urllib
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
from pydantic import BaseModel
from pyedge.models.base import BaseInferenceModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_imagenet_classes() -> List[str]:
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


class ClassificationConfig(BaseModel):
    model_name: str = "efficientnet_b0"
    num_outputs: int = 5

class Classification(BaseInferenceModel):
    def __init__(self, config: ClassificationConfig = ClassificationConfig()):
        super().__init__()
        self.config = config
        self.model = timm.create_model(self.config.model_name, pretrained=True).to(
            device
        )
        self.model.eval()
        self.classes = get_imagenet_classes()

        # Warm up
        self([np.random.rand(500, 500, 3).astype(np.float32) for _ in range(5)])

    def preprocess(self, images: List[np.ndarray]):
        images = torch.from_numpy(np.array(images)).to(device)
        return images.permute(0, -1, 1, 2)

    def postprocess(self, predictions: torch.Tensor):
        preds = nn.functional.softmax(predictions, dim=1).detach().cpu()
        return [
            sorted(
                list(zip(self.classes, np.array(im_pred))),
                key=lambda x: x[1],
                reverse=True,
            )[: self.config.num_outputs]
            for im_pred in preds
        ]

    def __call__(self, images: List[torch.Tensor]):
        with torch.no_grad():
            out = self.model(self.preprocess(images))
        return self.postprocess(out)


if __name__ == "__main__":
    model = Classification()
    num = 3
    ims = [np.random.rand(300, 300, 3).astype(np.float32) for _ in range(num)]
    import time

    start = time.time()
    preds = model(ims)
    end = time.time()
    run = end - start
    print(f"{round(run, 4)} seconds")
    print(f"{round(num / run, 4)} FPS")
    breakpoint()
