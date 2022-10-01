import logging
import multiprocessing as mp
from queue import Queue

logger = logging.getLogger(__name__)


class BaseProcessor:
    def __init__(self):
        self.manager_type = "Base"
        self.cache = 150

        self.ctx = None
        self.process = None

        self.in_q = None
        self.out_q = None

        self.stop_event = None
        self.start_event = None
        self.performance_tracker = []

    def start_process(
        self,
        in_q: Queue = None,
        out_q: Queue = None,
    ):
        self.in_q = in_q
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()

        self.out_q = out_q
        if self.out_q is None:
            self.out_q = self.ctx.Queue(maxsize=self.cache)

        self.process = self.ctx.Process(target=self.run_forever)

        self.process.daemon = True
        self.process.start()
        return self.out_q

    def _log_performance(self):
        avg_fps = round(
            sum(self.performance_tracker) / len(self.performance_tracker)
            if len(self.performance_tracker) > 0
            else 0,
            4,
        )
        logger.info(f"{self.manager_type}: Average FPS: {avg_fps}")

    def stop_process(self):
        self._close_objects()
        self.process.join(timeout=15.0)

        if self.process.is_alive():  # Ensure it's done
            logger.warning(f"{self.manager_type}: Backup termination used")
            self.process.terminate()
        logger.info(f"{self.manager_type}: Background process stopped")
        return True

    def _close_objects(self):
        if self.in_q is not None:
            self.in_q.close()
        if self.out_q is not None:
            self.out_q.close()
        if not self.stop_event.is_set():
            self.stop_event.set()
        if "stream" in self.manager_type.lower():
            for name, stream in self.streams.items():
                stream.release()
                logger.info(f"Stream: {name=} released")

    def _track_performance(self, rate):
        self.performance_tracker.append(rate)
        if len(self.performance_tracker) > 2000:
            self.performance_tracker = self.performance_tracker[-500:]

    def log(self, text: str, level: str):
        if level == "info":
            logger.info(text)
        elif level == "warning":
            logger.warning(text)
        elif level == "error":
            logger.error(text)

    def run_forever(self):
        """
        Here, the input start event is really only used to hold off on grabbing frames from the
        streams. OpenCV starts caching as soon as you open the connection so we want to wait until
        the model has warmed up before we open the connection.
        """
        pass

    def release(self):
        pass
