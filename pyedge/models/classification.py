import urllib
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn

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


class Classification:
    def __init__(self, model_name: str = "efficientnet_b0", num_outputs: int = 5):
        self.model_name = model_name
        self.num_outputs = num_outputs
        self.model = timm.create_model(self.model_name, pretrained=True).to(device)
        self.model.eval()
        self.classes = get_imagenet_classes()
        self.dtype = np.float32

        # Warm up
        for _ in range(5):
            _ = self([np.random.rand(500, 500, 3).astype(self.dtype)])

    async def preprocess(self, images: List[np.ndarray]):
        images = torch.from_numpy(np.asarray(images, dtype=self.dtype)).to(device)
        return images.permute(0, -1, 1, 2)

    async def postprocess(self, predictions: torch.Tensor):
        preds = nn.functional.softmax(predictions, dim=1).detach().cpu()
        return [
            sorted(
                list(zip(self.classes, np.asarray(im_pred))),
                key=lambda x: x[1],
                reverse=True,
            )[: self.num_outputs]
            for im_pred in preds
        ]

    async def __call__(self, images: List[torch.Tensor]):
        with torch.no_grad():
            out = self.model(await self.preprocess(images))
        return await self.postprocess(out)


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
