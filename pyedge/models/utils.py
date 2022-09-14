from typing import Optional

from pydantic import BaseModel


class ClassificationConfig(BaseModel):
    model_name: str = "efficientnet_b0"
    num_outputs: int = 5


class DetectionConfig(BaseModel):
    model_name: str = "faster_rcnn"
    threshold: float = 0.5


class ModelConfig(BaseModel):
    classification: Optional[ClassificationConfig] = ClassificationConfig
    detection: Optional[DetectionConfig] = None
    max_batch_size: int = 16
