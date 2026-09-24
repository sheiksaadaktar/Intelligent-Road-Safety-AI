"""
Runtime health monitoring for the Intelligent Road Safety AI system.

This module observes runtime behavior without modifying the safety
decision logic. It provides operational diagnostics for input,
processing, alerts, incidents, and output state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RuntimeHealth:
    """Track runtime health independently from safety decisions."""

    source_opened: bool = False

    frames_read: int = 0
    frame_read_failures: int = 0
    frames_processed: int = 0
    frames_with_detections: int = 0

    confirmed_alerts: int = 0
    total_incidents: int = 0
    resolved_incidents: int = 0

    processing_time_seconds: float = 0.0

    started_at: float = field(
        default_factory=time.perf_counter
    )

    finished_at: Optional[float] = None

    final_state: str = "INITIALIZING"

    def mark_source_opened(self, opened: bool) -> None:
        """Record whether the input source opened successfully."""

        self.source_opened = bool(opened)

        if self.source_opened:
            self.final_state = "RUNNING"
        else:
            self.final_state = "INPUT_ERROR"

    def record_frame_read(self, success: bool) -> None:
        """Record the result of a frame read."""

        if success:
            self.frames_read += 1
        else:
            self.frame_read_failures += 1

    def record_frame_processed(
        self,
        has_detections: bool,
    ) -> None:
        """Record a successfully processed frame."""

        self.frames_processed += 1

        if has_detections:
            self.frames_with_detections += 1

    def record_processing_time(
        self,
        elapsed_seconds: float,
    ) -> None:
        """Accumulate processing time for a frame."""

        if elapsed_seconds >= 0:
            self.processing_time_seconds += elapsed_seconds

    def update_alert_count(
        self,
        confirmed_alerts: int,
    ) -> None:
        """Update the number of confirmed alerts."""

        self.confirmed_alerts = max(
            0,
            int(confirmed_alerts),
        )

    def update_incident_counts(
        self,
        total_incidents: int,
        resolved_incidents: int,
    ) -> None:
        """Update incident lifecycle counts."""

        self.total_incidents = max(
            0,
            int(total_incidents),
        )

        self.resolved_incidents = max(
            0,
            int(resolved_incidents),
        )

    def finish(self) -> None:
        """Mark the runtime as completed."""

        self.finished_at = time.perf_counter()

        if self.final_state not in {
            "INPUT_ERROR",
            "PROCESSING_ERROR",
        }:
            self.final_state = "COMPLETED"

    @property
    def elapsed_seconds(self) -> float:
        """Return total wall-clock runtime."""

        end_time = (
            self.finished_at
            if self.finished_at is not None
            else time.perf_counter()
        )

        return max(
            0.0,
            end_time - self.started_at,
        )

    @property
    def average_processing_fps(self) -> float:
        """Return average processing FPS based on measured work time."""

        if self.processing_time_seconds <= 0:
            return 0.0

        return (
            self.frames_processed
            / self.processing_time_seconds
        )

    @property
    def detection_rate_percent(self) -> float:
        """Return percentage of processed frames containing detections."""

        if self.frames_processed <= 0:
            return 0.0

        return (
            self.frames_with_detections
            / self.frames_processed
            * 100.0
        )

    def summary(self) -> Dict[str, object]:
        """Return a JSON-compatible runtime health summary."""

        return {
            "state": self.final_state,
            "source_opened": self.source_opened,
            "frames_read": self.frames_read,
            "frame_read_failures": self.frame_read_failures,
            "frames_processed": self.frames_processed,
            "frames_with_detections": (
                self.frames_with_detections
            ),
            "detection_rate_percent": round(
                self.detection_rate_percent,
                2,
            ),
            "confirmed_alerts": self.confirmed_alerts,
            "total_incidents": self.total_incidents,
            "resolved_incidents": self.resolved_incidents,
            "processing_time_seconds": round(
                self.processing_time_seconds,
                4,
            ),
            "elapsed_seconds": round(
                self.elapsed_seconds,
                4,
            ),
            "average_processing_fps": round(
                self.average_processing_fps,
                2,
            ),
        }
