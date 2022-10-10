import argparse
from imp import reload
import logging
from typing import List, Optional

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from pyedge.model import ModelProcessor
from pyedge.models.utils import ModelConfig
from pyedge.stream import Stream, StreamConfig, StreamProcessor

logger = logging.getLogger(__name__)

app = FastAPI()
deployments = {}
current_deployment = None

class DeploymentRequest(BaseModel):
    name: str
    stream_config: StreamConfig
    model_config: ModelConfig
    activate: bool = False
    overwrite: bool = True


class Deployment(DeploymentRequest, extra='allow'):
    _running: Optional[bool] = False
    _stream: Optional[StreamProcessor] = None
    _model: Optional[ModelProcessor] = None

    def dict(self, *args, **kwargs):
        return {
            'name': self.name, 
            'stream_config': self.stream_config, 
            'model_config': self.model_config, 
            'active': self._running,
        }

    def start(self):
        self._stream = StreamProcessor(self.stream_config)
        self._model = ModelProcessor(self.model_config)
        stream_q = self._stream.start_process()
        model_q = self._model.start_process(in_q=stream_q)
        self._running = True

    def stop(self):
        self._stream.stop_process()
        self._model.stop_process()
        self._running = False


@app.post("/add")
async def add(deployment: DeploymentRequest):
    deployment = Deployment(**deployment.dict())
    if deployment.name in deployments.keys() and not deployment.overwrite:
        return {'error': f"{deployment.name} already exists"}
    deployments[deployment.name] = deployment
    message = f'{deployment.name} added to deployments'
    if current_deployment is None and deployment.activate:
        logger.info("Starting deployment")
        deployment.start()
        message = f'{deployment.name} added and started'
    return {'message': message}


@app.post("/start")
async def start(deployment: str):
    pass

@app.put('/stop')
async def stop(deployment: str):
    dep = deployments.get(deployment, None)
    if dep is not None:
        dep.stop()
        message = f"{deployment} stopped"
    else:
        message = f"{deployment} not found"
    current_deployment = None
    return {'message': message}

@app.get("/deployments")
async def get_deployments():
    return deployments

if __name__ == "__main__":
    uvicorn.run("pyedge.main:app", port=8888, log_level="info", reload=True)
