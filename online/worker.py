from collections import deque
from dataclasses import dataclass
import threading
import numpy as np

@dataclass
class OnlineWindow:
    generation: int
    number: int
    indices: np.ndarray
    observations: np.ndarray

class OnlineWorker:

    def __init__(self, model):
        self.model = model
        self.condition = threading.Condition()
        self.pending = None
        self.results = deque()
        self.closing = False
        self.error = None
        self.dropped_windows = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, window):
        with self.condition:
            if self.closing:
                raise RuntimeError("Online inference worker is closed")
            if self.pending is not None:
                self.dropped_windows += 1
            self.pending = window
            self.condition.notify()

    def reset(self):
        with self.condition:
            self.pending = None
            self.results.clear()

    def _loop(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closing or self.pending is not None)
                    if self.pending is None:
                        return
                    window, self.pending = self.pending, None
                result = self.model.predict_online(window)
                with self.condition:
                    self.results.append((window, result))
        except Exception as exc:
            self.error = exc

    def drain(self):
        if self.error is not None:
            raise RuntimeError("Online inference failed") from self.error
        with self.condition:
            result = list(self.results)
            self.results.clear()
            return result

    def close(self):
        with self.condition:
            self.closing = True
            self.condition.notify()
        self.thread.join(10)
        if self.thread.is_alive():
            raise RuntimeError("Timed out finishing online inference")
