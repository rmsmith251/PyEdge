from typing import Optional

from pydantic import BaseModel
from pyedge.models.classification import Classification, ClassificationConfig
from pyedge.models.detection import ObjectDetection, DetectionConfig


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
