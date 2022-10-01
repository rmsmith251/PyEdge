import argparse
import logging
from typing import List, Optional

from pydantic import BaseModel

from pyedge.model import ModelProcessor
from pyedge.models import ModelConfig
from pyedge.stream import Stream, StreamConfig, StreamProcessor

logger = logging.getLogger(__name__)


class Deployment(BaseModel):
    stream_config: StreamConfig
    model_config: ModelConfig
    _stream: Optional[StreamProcessor]
    _model: Optional[ModelProcessor]

    def start(self):
        self._stream = StreamProcessor(self.stream_config)
        self._model = ModelProcessor(self.model_config)
        stream_q = self._stream.start_process()
        model_q = self._model.start_process(in_q=stream_q)

    def stop(self):
        self._stream.stop_process()
        self._model.stop_process()


def deploy(args):
    if args.config:
        dep = Deployment(**args.config)
    elif args.stream_url:
        dep = Deployment(
            stream_config=StreamConfig(
                streams=[Stream(url=args.stream_url, name=args.stream_url)]
            )
        )
    logger.info("Starting deployment")
    dep.start()
    while True:
        try:
            ### TODO: Get metrics periodically
            pass
        except KeyboardInterrupt:
            logger.info("Shutting down")
            dep.stop()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="A JSON config file defining the stream and model properties",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default="",
        help="Used to initialize a default deployment",
    )
    args = parser.parse_args()
    deploy(args)
