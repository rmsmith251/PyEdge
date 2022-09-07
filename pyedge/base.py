import logging
import multiprocessing as mp
from queue import Queue
from threading import Event


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
        make_start_event: bool = True,
        input_start_event: Event = None,
    ):
        self.in_q = in_q
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        if make_start_event:
            self.start_event = self.ctx.Event()
        elif input_start_event is not None:
            self.start_event = input_start_event

        self.out_q = out_q
        if self.out_q is None:
            self.out_q = self.ctx.Queue(maxsize=self.cache)

        self.process = self.ctx.Process(
            target=self.run_forever, args=(input_start_event,)
        )

        self.process.daemon = True
        self.process.start()
        return self.out_q, self.start_event

    def _log_performance(self):
        avg_fps = round(
            sum(self.performance_tracker) / len(self.performance_tracker)
            if len(self.performance_tracker) > 0
            else 0,
            4,
        )
        logging.info(f"{self.manager_type}: Average FPS: {avg_fps}")

    def stop_process(self):
        self._close_objects()
        self.process.join(timeout=15.0)

        if self.process.is_alive():  # Ensure it's done
            logging.warning(f"{self.manager_type}: Backup termination used")
            self.process.terminate()
        logging.info(f"{self.manager_type}: Background process stopped")
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
                logging.info(f"Stream: {name=} released")

    def _track_performance(self, rate):
        self.performance_tracker.append(rate)
        if len(self.performance_tracker) > 2000:
            self.performance_tracker = self.performance_tracker[-500:]

    def run_forever(self, input_start_event: Event = None):
        pass

    def release(self):
        pass
