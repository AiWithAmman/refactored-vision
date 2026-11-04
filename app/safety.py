"""Keep a failed model request from crashing the demo or exposing provider details."""

import logging

log = logging.getLogger("lumen")
FRIENDLY = "I couldn't reach the model just now. Please try again."


def safely(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        log.exception("Lumen tool failed")
        return {"say": FRIENDLY, "error": type(exc).__name__}
