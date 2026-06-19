import time
from threading import Lock

class PerModelRateLimiter:
    """
    Separate rate limiter per model to respect individual RPM limits.
    """
    def __init__(self):
        self._limiters: dict[str, "_Limiter"] = {}
        self._lock = Lock()

    def wait(self, model_id: str, rpm: int):
        with self._lock:
            if model_id not in self._limiters:
                self._limiters[model_id] = _Limiter(rpm)
        self._limiters[model_id].wait()


class _Limiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.last = 0.0
        self._lock = Lock()

    def wait(self):
        with self._lock:
            elapsed = time.monotonic() - self.last
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last = time.monotonic()


# Global singleton
rate_limiter = PerModelRateLimiter()
