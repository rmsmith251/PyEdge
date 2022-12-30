from typing import List

from fastapi import UploadFile
from pydantic import BaseModel, BaseSettings


class HealthResponse(BaseModel):
    message: str = "ok"


class InferenceRequest(BaseModel):
    images: List[UploadFile]


class InferenceResponse(BaseModel):
    data: List


class ModelPipeline(BaseModel):
    models: List
