import asyncio
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
from pyedge.api.types import HealthResponse

app = FastAPI()
settings = BaseSettings()


@app.get("/health")
async def health():
    return HealthResponse()


@app.post("/add")
async def create():
    return {}


@app.post("/{deployment_id}/delete")
async def delete(deployment_id: str):
    return {}


@app.post("/{deployment_id}/start")
async def start(deployment_id: str):
    return {}


@app.post("/{deployment_id}/stop")
async def stop(deployment_id: str):
    return {}
