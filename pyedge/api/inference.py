import asyncio

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pyedge.api.types import HealthResponse, InferenceRequest, InferenceResponse
from pyedge.utils import bytes_to_cv2

app = FastAPI()
model = None


@app.get("/health", response_format=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/create")
async def create():
    return JSONResponse(content={}, status_code=200)


@app.post("/predict", response_format=InferenceResponse)
async def predict(images: InferenceRequest):
    img_bytes = await asyncio.gather(*[image_bytes.read() for image_bytes in images])
    images = await asyncio.gather(*[bytes_to_cv2(bytes_) for bytes_ in img_bytes])
    return asyncio.run(model(images))
