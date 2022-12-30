from typing import Literal, Optional, Union

from pydantic import BaseModel
from pyedge.models.classification import Classification
from pyedge.models.detection import ObjectDetection


class ClassificationModel(BaseModel):
    type: Literal["classification"] = "classification"
    model_name: str = "efficientnet_b0"
    num_outputs: int = 5


class DetectionModel(BaseModel):
    type: Literal["detection"] = "detection"
    model_name: str = "faster_rcnn"
    threshold: float = 0.5
    _model: Optional[ObjectDetection]

    def __init__(self, **data):
        super().__init__(**data)
        self._model = ObjectDetection()


class TwoStageModel(BaseModel):
    type: Literal["two-stage"] = "two-stage"
    classification_model_name: str = "efficientnet_b0"
    detection_model_name: str = "faster_rcnn"
    num_outputs: int = 5
    threshold: float = 0.5
    crop_class: Optional[str] = None


model = Union[ClassificationModel, DetectionModel, TwoStageModel]
