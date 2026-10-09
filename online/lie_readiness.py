import numpy as np
from .lie_settling import PROTOCOL, QuietGate

class LieReadiness:
    def __init__(self, dt, baseline_s):
        self.gate = QuietGate(dt)
        self.baseline_s = float(baseline_s)
        self.settled_at = self.ready_at = self.first_force_at = None
        self.timed_out = False
        self.last_speeds = np.zeros(3)

    @property
    def ready(self):
        return self.ready_at is not None and not self.timed_out

    def update(self, elapsed, speeds, quiet):
        self.last_speeds = np.asarray(speeds).copy()
        if self.first_force_at is not None:
            return
        if not quiet:
            self.settled_at = self.ready_at = None
        stable = self.gate.update(elapsed, speeds, quiet)
        if stable and self.settled_at is None:
            self.settled_at = float(elapsed)
        if self.settled_at is None and elapsed >= PROTOCOL["timeout_s"]:
            self.timed_out = True
        if self.settled_at is not None and elapsed - self.settled_at >= self.baseline_s - 1e-9:
            if self.ready_at is None:
                self.ready_at = float(elapsed)
                self.timed_out = False

    def start_force(self, elapsed):
        if not self.ready:
            raise RuntimeError("Lie force requested before settled baseline completed")
        if self.first_force_at is None:
            self.first_force_at = float(elapsed)

    def status(self, elapsed):
        if self.ready:
            return "READY - select a body and pull upward"
        if self.settled_at is not None:
            return f"QUIET BASELINE {max(0.0, self.baseline_s - (elapsed - self.settled_at)):.1f} s"
        if self.timed_out:
            return "STILL SETTLING (>20 s) - wait or R to retry"
        return f"SETTLING - quiet {self.gate.quiet_s:.1f}/1.0 s"

    def metadata(self):
        return dict(
            settled_at_s=self.settled_at,
            ready_at_s=self.ready_at,
            first_force_at_s=self.first_force_at,
            baseline_s=self.baseline_s,
            timed_out=self.timed_out,
            final_speeds=self.last_speeds.tolist(),
        )
