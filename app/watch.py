"""Heuristic approaching-object alerts from successive object boxes."""

from threading import Lock
from typing import Dict, Optional, Tuple

from .guidance import CLOSE, Box

MIN_AREA = 0.05
MIN_GROWTH_PER_S = 1.3
MIN_DT, MAX_DT = 0.25, 4.0


def approach_alert(
    concept: str, prev_area: Optional[float], box: Box, dt: float
) -> Optional[dict]:
    if (
        prev_area is None
        or prev_area <= 0
        or dt <= 0
        or dt > MAX_DT
        or box.area < MIN_AREA
    ):
        return None
    growth = (box.area / prev_area) ** (1 / max(dt, MIN_DT))
    if growth < MIN_GROWTH_PER_S:
        return None
    side = "from your left" if box.cx < 0.35 else "from your right" if box.cx > 0.65 else "ahead of you"
    close = box.area >= CLOSE
    say = f"{concept.capitalize()} approaching {side}."
    if close:
        say = f"Careful. {say} Very close."
    return {"concept": concept, "close": close, "say": say}


class Tracker:
    """Keep the previous box size for each watched object and browser session."""

    def __init__(self) -> None:
        self._last: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._lock = Lock()

    def update(
        self, session: str, concept: str, box: Optional[Box], now: float
    ) -> Optional[dict]:
        key = (session, concept)
        with self._lock:
            previous = self._last.get(key)
            if box is None:
                self._last.pop(key, None)
                return None
            self._last[key] = (now, box.area)
        if previous is None:
            return None
        return approach_alert(concept, previous[1], box, now - previous[0])
