"""Turns a SAM 3.1 bounding box into a short spoken cue.

Pure functions, no network. This is the 'last ten feet' part of Lumen.
It is an aid and does not replace a cane or guide dog.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2 / self.w

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2 / self.h

    @property
    def area(self) -> float:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1)) / (self.w * self.h)


LEFT, RIGHT = 0.35, 0.65
NEAR, CLOSE = 0.08, 0.25


def cue_for(box: Optional[Box]) -> dict:
    if box is None:
        return {
            "found": False,
            "done": False,
            "say": "I can't see it yet. Turn slowly to your left or right.",
        }

    info = {
        "found": True,
        "cx": round(box.cx, 2),
        "cy": round(box.cy, 2),
        "area": round(box.area, 3),
    }

    if box.cx < LEFT:
        return {**info, "done": False, "say": "Turn a little to your left."}
    if box.cx > RIGHT:
        return {**info, "done": False, "say": "Turn a little to your right."}

    if box.area >= CLOSE:
        if box.cy < 0.35:
            hint = "Reach up."
        elif box.cy > 0.65:
            hint = "Reach down."
        else:
            hint = "Reach straight out in front of you."
        return {**info, "done": True, "say": f"It's right in front of you. {hint}"}

    if box.area >= NEAR:
        return {**info, "done": False, "say": "Getting close. Keep going slowly."}

    return {**info, "done": False, "say": "It's straight ahead. Keep walking forward."}
