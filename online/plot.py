import mujoco
import numpy as np

class OnlinePlot:
    def __init__(self, seconds=8.0, component="force"):
        if component not in ("force", "torque"):
            raise ValueError("Choose force or torque for the plot")
        self.component = component

        self.unit = "N" if component == "force" else "N.m"
        self.symbol = "F" if component == "force" else "T"
        self.seconds = seconds
        self.title = "ONLINE | sixthsense | fixed-lag estimates"
        self.figure = mujoco.MjvFigure()
        f = self.figure
        f.title = f"True and estimated {component} magnitude ({self.unit})"
        f.xlabel = "simulation time (s) - predictions aligned to their target time"
        f.flg_legend = True
        f.flg_extend = False

        f.flg_barplot = True
        f.gridsize[:] = [6, 5]
        f.linewidth = 3.0
        f.gridrgb[:] = [0.20, 0.25, 0.31]
        f.figurergba[:] = [0.055, 0.075, 0.10, 1.0]
        f.panergba[:] = [0.075, 0.10, 0.13, 1.0]
        f.legendrgba[:] = [0.055, 0.075, 0.10, 0.95]
        f.textrgb[:] = [0.90, 0.94, 0.98]
        f.xformat, f.yformat = "%.1f", "%.0f" if component == "force" else "%.2f"

        f.linename[0] = f"Estimated |{self.symbol}| ({self.unit})"
        f.linename[1] = f"True |{self.symbol}| ({self.unit})"
        f.linergb[0] = [0.12, 0.95, 0.42]
        f.linergb[1] = [1.0, 0.32, 0.29]

    def segments(self, x, y, max_gap=None, marker_half_width=None, max_points=None):

        if not len(x):
            return np.empty((0, 2))

        gaps = (
            np.r_[0, np.cumsum(np.diff(x) > max_gap + 1e-7)]
            if max_gap is not None
            else np.zeros(len(x))
        )
        capacity = self.figure.linedata.shape[1] // 4 + (marker_half_width is None)
        max_points = capacity if max_points is None else min(capacity, max_points)
        if len(x) > max_points:
            selected = [0, len(x) - 1]
            for group in np.array_split(np.arange(1, len(x) - 1), (max_points - 2) // 2):
                selected.extend([group[np.argmin(y[group])], group[np.argmax(y[group])]])
            selected = np.unique(selected)
            x, y, gaps = x[selected], y[selected], gaps[selected]
        points = np.column_stack([x, y])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        connected = np.diff(gaps) == 0
        segments = segments[connected]
        if marker_half_width is not None:
            isolated = np.ones(len(points), bool)
            indices = np.flatnonzero(connected)
            isolated[indices] = False
            isolated[indices + 1] = False
            if isolated.any():
                markers = np.repeat(points[isolated, None, :], 2, axis=1)
                markers[:, 0, 0] -= marker_half_width
                markers[:, 1, 0] += marker_half_width
                segments = np.concatenate([segments, markers])
        return segments.reshape(-1, 2)

    def dashed_segments(self, x, y):

        if len(x) < 2:
            return self.segments(x, y, marker_half_width=0.01)
        period = self.seconds / 64
        dash = period * 0.6

        capacity = self.figure.linedata.shape[1] // 4
        periods = int(np.ceil((x[-1] - x[0]) / period)) + 2
        budget = max(4, capacity - 2 * periods)
        line = self.segments(x, y, max_points=budget).reshape(-1, 2, 2)
        result = []
        for (a, fa), (b, fb) in line:
            if b <= a:
                continue
            for cycle in range(int(np.floor(a / period)), int(np.floor(b / period)) + 1):
                left, right = max(a, cycle * period), min(b, cycle * period + dash)
                if right <= left:
                    continue
                result.extend(
                    [
                        (left, fa + (fb - fa) * (left - a) / (b - a)),
                        (right, fa + (fb - fa) * (right - a) / (b - a)),
                    ]
                )
        return np.asarray(result).reshape(-1, 2)

    @staticmethod
    def viewport(viewport, side=None):
        width = viewport.width if side is None else viewport.width // 2
        left = viewport.left if side in (None, 0) else viewport.left + width
        if side == 1:
            width = viewport.width - width
        return mujoco.MjrRect(
            left, viewport.bottom, width, max(1, int(viewport.height * 0.26))
        )

    def update_live(self, now, generation, predictions, samples):
        f = self.figure
        f.linepnt[:] = 0
        start = max(0.0, now - self.seconds)
        truth = [
            s
            for gen, s in samples[-int(self.seconds * 50) - 4 :]
            if gen == generation and start <= s.time <= now
        ]
        rows = [
            r
            for r in predictions[-int(self.seconds * 50) - 200 :]
            if r["generation"] == generation and start <= r["target_time"] <= now
        ]
        series = [
            (np.array([s.time for s in truth]), np.array([np.linalg.norm(getattr(s, self.component)) for s in truth])),
            (
                np.array([r["target_time"] for r in rows]),
                np.array([np.linalg.norm(r[self.component]) for r in rows]),
            ),
        ]
        ymax = 20.0 if self.component == "force" else 2.0
        for i, (times, forces) in enumerate(series):
            points = (
                self.segments(times, forces, max_gap=0.021)
                if i == 0
                else self.dashed_segments(times, forces)
            )
            line = 1 - i
            f.linepnt[line] = len(points)
            f.linedata[line, : 2 * len(points)] = points.ravel()
            if len(forces):
                ymax = max(ymax, float(forces.max()) * 1.2)
        f.range[0] = [start, max(self.seconds, now)]

        f.range[1] = [-0.04 * ymax, ymax]
        f.xlabel = f"simulation time (s); true and estimated {self.component} at the SAME target time"
        f.title = self.title
        if rows:
            row = rows[-1]
            digits = 1 if self.component == "force" else 2
            f.title += f" | t={row['target_time']:.2f} True {np.linalg.norm(row['true_' + self.component]):.{digits}f} / Est {np.linalg.norm(row[self.component]):.{digits}f} {self.unit}"
        else:
            f.title += " | waiting for robot observations"
        return f
