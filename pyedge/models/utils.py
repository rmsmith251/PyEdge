from typing import Optional

from pydantic import BaseModel
from pyedge.models.classification import Classification
from pyedge.models.detection import ObjectDetection


class ClassificationConfig(BaseModel):
    model_name: str = "efficientnet_b0"
    num_outputs: int = 5


class DetectionConfig(BaseModel):
    model_name: str = "faster_rcnn"
    threshold: float = 0.5


class ModelConfig(BaseModel):
    classification: Optional[ClassificationConfig] = None
    detection: Optional[DetectionConfig] = None
    max_batch_size: int = 16

    def get_model(self):
        if self.classification is not None:
            return Classification(config=self.classification)
        elif self.detection is not None:
            return ObjectDetection(config=self.detection)
        else:
            return Classification(config=ClassificationConfig())
