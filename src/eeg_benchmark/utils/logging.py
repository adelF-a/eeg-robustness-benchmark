"""Standard-library logging configuration for the benchmark."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "eeg_benchmark") -> logging.Logger:
    """Return a configured logger, initialising root handlers once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        # MNE is very chatty; keep warnings and errors only.
        logging.getLogger("mne").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
