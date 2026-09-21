"""
Central configuration for the Intelligent Road Safety AI system.

This module centralizes runtime parameters that were previously
defined across individual processing components.

The initial values intentionally match the existing production
configuration so that behavior remains unchanged.
"""


from pathlib import Path


# ============================================================
# MODEL / INPUT CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

VIDEO_PATH = "data/tracking_test.mp4"


# ============================================================
# RISK DISTANCE THRESHOLDS
# ============================================================

MAX_RISK_DISTANCE = 320

CRITICAL_DISTANCE = 100
HIGH_DISTANCE = 170
MEDIUM_DISTANCE = 260


# ============================================================
# PREDICTED DISTANCE THRESHOLDS
# ============================================================

CRITICAL_PREDICTED_DISTANCE = 30
HIGH_PREDICTED_DISTANCE = 70
MEDIUM_PREDICTED_DISTANCE = 150


# ============================================================
# APPROACH SPEED THRESHOLDS
# ============================================================

MIN_APPROACH_SPEED = 20
HIGH_APPROACH_SPEED = 80
CRITICAL_APPROACH_SPEED = 160


# ============================================================
# TIME-TO-COLLISION THRESHOLDS
# ============================================================

CRITICAL_TCA = 0.6
HIGH_TCA = 1.2
MEDIUM_TCA = 2.5
MAX_TCA = 5.0


# ============================================================
# CONVERGENCE THRESHOLDS
# ============================================================

MIN_CONVERGENCE = 0.30
STRONG_CONVERGENCE = 0.75


# ============================================================
# VELOCITY / TRACKING SAFETY LIMITS
# ============================================================

HISTORY_SIZE = 5

MAX_REASONABLE_VELOCITY = 800

MAX_REASONABLE_RELATIVE_VELOCITY = 1600


# ============================================================
# EVENT GROUPING
# ============================================================

MIN_EVENT_FRAMES = 5

EVENT_GAP_TOLERANCE = 8

# At approximately 59.94 FPS, 15 frames is about 0.25 seconds.
EVENT_SAFE_GAP_FRAMES = 15


# ============================================================
# EVENT CONFIRMATION
# ============================================================

MIN_STRONG_OBSERVATIONS = 3

STRONG_TCA_LIMIT = 1.5

STRONG_PREDICTED_DISTANCE = 80

STRONG_APPROACH_SPEED = 80

MIN_STRONG_SPAN_FRAMES = 4

MIN_STRONG_FRAME_GAP = 2


# ============================================================
# LIVE ALERT OUTPUT
# ============================================================

LIVE_ALERT_OUTPUT_PATH = Path(
    "output/live_alerts.json"
)


# ============================================================
# LIVE INCIDENT OUTPUT
# ============================================================

LIVE_INCIDENT_OUTPUT_PATH = Path(
    "output/live_incidents.json"
)


# ============================================================
# SAFETY ANALYTICS OUTPUT
# ============================================================

SAFETY_ANALYTICS_OUTPUT_PATH = Path(
    "output/safety_analytics.json"
)


# ============================================================
# SAFETY EVALUATION OUTPUT
# ============================================================

SAFETY_EVALUATION_OUTPUT_PATH = Path(
    "output/safety_evaluation.json"
)


# ============================================================
# UI CONFIGURATION
# ============================================================

WINDOW_NAME = (
    "Intelligent Road Safety AI - "
    "Real-Time Safety Monitor"
)

DISPLAY_WIDTH = 1280

DISPLAY_HEIGHT = 720

UI_SCALE = 1.45

MAX_DISPLAY_PAIRS = 5

MAX_INCIDENT_HISTORY = 5