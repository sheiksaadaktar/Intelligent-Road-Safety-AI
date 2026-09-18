from pathlib import Path
import json
import cv2
import numpy as np

from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/tracking_test.mp4"


# ============================================================
# CURRENT DISTANCE
# ============================================================

MAX_RISK_DISTANCE = 320

CRITICAL_DISTANCE = 100
HIGH_DISTANCE = 170
MEDIUM_DISTANCE = 260


# ============================================================
# PREDICTED CLOSEST DISTANCE
# ============================================================

CRITICAL_PREDICTED_DISTANCE = 30
HIGH_PREDICTED_DISTANCE = 70
MEDIUM_PREDICTED_DISTANCE = 150


# ============================================================
# APPROACH SPEED
# ============================================================

MIN_APPROACH_SPEED = 20
HIGH_APPROACH_SPEED = 80
CRITICAL_APPROACH_SPEED = 160


# ============================================================
# TIME TO CLOSEST APPROACH
# ============================================================

CRITICAL_TCA = 0.6
HIGH_TCA = 1.2
MEDIUM_TCA = 2.5
MAX_TCA = 5.0


# ============================================================
# CONVERGENCE
# ============================================================

MIN_CONVERGENCE = 0.30
STRONG_CONVERGENCE = 0.75


# ============================================================
# VELOCITY
# ============================================================

HISTORY_SIZE = 5

MAX_REASONABLE_VELOCITY = 800
MAX_REASONABLE_RELATIVE_VELOCITY = 1600


# ============================================================
# EVENT GROUPING
# ============================================================

MIN_EVENT_FRAMES = 5
EVENT_GAP_TOLERANCE = 8

# V5.2: end a risk episode after sustained LOW risk.
# At ~59.94 FPS, 15 frames ~= 0.25 seconds.
EVENT_SAFE_GAP_FRAMES = 15


# ============================================================
# V5.1 CONFIRMATION
# ============================================================

MIN_STRONG_OBSERVATIONS = 3

STRONG_TCA_LIMIT = 1.5
STRONG_PREDICTED_DISTANCE = 80
STRONG_APPROACH_SPEED = 80

# Prevent three strong observations from being counted when
# they are all concentrated into one or two frames.
MIN_STRONG_SPAN_FRAMES = 4

# Strong observations must have some temporal separation.
MIN_STRONG_FRAME_GAP = 2


# ============================================================
# HELPERS
# ============================================================

def bottom_center(box):
    """
    Return the bottom-center of a bounding box.

    This is more appropriate than the geometric center for
    road-scene interaction because it approximates the point
    where the object contacts the road.
    """
    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2.0,
            y2,
        ],
        dtype=np.float32,
    )


def smooth_velocity(history):
    """
    Average recent velocity vectors.
    """
    if not history:
        return np.zeros(2, dtype=np.float32)

    return np.mean(
        np.asarray(history, dtype=np.float32),
        axis=0,
    )


def calculate_trajectory_metrics(
    position_1,
    velocity_1,
    position_2,
    velocity_2,
):
    """
    Calculate relative trajectory metrics for two objects.

    Returns:
        distance
        relative_speed
        approach_speed
        convergence
        tca
        predicted_distance
        converging
    """

    position_1 = np.asarray(position_1, dtype=np.float32)
    position_2 = np.asarray(position_2, dtype=np.float32)
    velocity_1 = np.asarray(velocity_1, dtype=np.float32)
    velocity_2 = np.asarray(velocity_2, dtype=np.float32)

    # --------------------------------------------------------
    # Relative position
    # --------------------------------------------------------

    r = position_2 - position_1

    # --------------------------------------------------------
    # Relative velocity
    # --------------------------------------------------------

    v = velocity_2 - velocity_1

    distance_now = float(np.linalg.norm(r))
    relative_speed = float(np.linalg.norm(v))

    # --------------------------------------------------------
    # Same-position safety case
    # --------------------------------------------------------

    if distance_now <= 1e-6:
        return {
            "distance": 0.0,
            "relative_speed": relative_speed,
            "approach_speed": relative_speed,
            "convergence": 1.0,
            "tca": 0.0,
            "predicted_distance": 0.0,
            "converging": True,
        }

    # --------------------------------------------------------
    # Radial approach speed
    # --------------------------------------------------------

    r_unit = r / distance_now

    approach_speed = -float(np.dot(r_unit, v))

    # --------------------------------------------------------
    # Convergence
    #
    # +1 = directly approaching
    #  0 = perpendicular
    # -1 = directly separating
    # --------------------------------------------------------

    if relative_speed > 1e-6:
        convergence = approach_speed / relative_speed
        convergence = float(
            np.clip(convergence, -1.0, 1.0)
        )
    else:
        convergence = 0.0

    # --------------------------------------------------------
    # Time to Closest Approach
    # --------------------------------------------------------

    velocity_squared = float(np.dot(v, v))

    if velocity_squared <= 1e-6:
        tca = float("inf")
        predicted_distance = distance_now

    else:
        tca = -float(np.dot(r, v)) / velocity_squared

        if tca < 0:
            tca = 0.0

        closest_position = r + (v * tca)

        predicted_distance = float(
            np.linalg.norm(closest_position)
        )

    # --------------------------------------------------------
    # Genuine future convergence
    # --------------------------------------------------------

    converging = (
        approach_speed >= MIN_APPROACH_SPEED
        and convergence >= MIN_CONVERGENCE
        and tca <= MAX_TCA
    )

    return {
        "distance": distance_now,
        "relative_speed": relative_speed,
        "approach_speed": approach_speed,
        "convergence": convergence,
        "tca": tca,
        "predicted_distance": predicted_distance,
        "converging": converging,
    }


# ============================================================
# V5 RISK CALCULATION
# ============================================================

def calculate_risk(metrics):
    """
    V5 trajectory-aware risk model.

    Risk uses:
        1. Current distance
        2. Predicted closest distance
        3. Approach speed
        4. Time to closest approach
        5. Convergence

    Risk is first gated by genuine convergence.
    """

    distance = metrics["distance"]
    predicted_distance = metrics["predicted_distance"]
    approach_speed = metrics["approach_speed"]
    tca = metrics["tca"]
    convergence = metrics["convergence"]
    converging = metrics["converging"]

    # --------------------------------------------------------
    # Basic gating
    # --------------------------------------------------------

    if distance > MAX_RISK_DISTANCE:
        return "LOW", 0

    if not converging:
        return "LOW", 0

    # --------------------------------------------------------
    # Score
    #
    # Maximum = 120
    # --------------------------------------------------------

    score = 0

    # Current distance
    if distance <= CRITICAL_DISTANCE:
        score += 30
    elif distance <= HIGH_DISTANCE:
        score += 20
    elif distance <= MEDIUM_DISTANCE:
        score += 10

    # Predicted closest distance
    if predicted_distance <= CRITICAL_PREDICTED_DISTANCE:
        score += 30
    elif predicted_distance <= HIGH_PREDICTED_DISTANCE:
        score += 25
    elif predicted_distance <= MEDIUM_PREDICTED_DISTANCE:
        score += 15

    # Approach speed
    if approach_speed >= CRITICAL_APPROACH_SPEED:
        score += 20
    elif approach_speed >= HIGH_APPROACH_SPEED:
        score += 15
    elif approach_speed >= MIN_APPROACH_SPEED:
        score += 5

    # TCA
    if tca <= CRITICAL_TCA:
        score += 25
    elif tca <= HIGH_TCA:
        score += 20
    elif tca <= MEDIUM_TCA:
        score += 10

    # Convergence
    if convergence >= STRONG_CONVERGENCE:
        score += 15
    elif convergence >= 0.60:
        score += 8
    else:
        score += 3

    # --------------------------------------------------------
    # CRITICAL trajectory conditions
    # --------------------------------------------------------

    critical_trajectory = (
        (
            predicted_distance <= CRITICAL_PREDICTED_DISTANCE
            and tca <= CRITICAL_TCA
        )
        or
        (
            distance <= CRITICAL_DISTANCE
            and approach_speed >= CRITICAL_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
        or
        (
            predicted_distance <= CRITICAL_PREDICTED_DISTANCE
            and approach_speed >= CRITICAL_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
        or
        (
            tca <= 0.30
            and approach_speed >= CRITICAL_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
    )

    if critical_trajectory and score >= 70:
        return "CRITICAL", score

    # --------------------------------------------------------
    # HIGH trajectory conditions
    # --------------------------------------------------------

    high_trajectory = (
        (
            predicted_distance <= HIGH_PREDICTED_DISTANCE
            and tca <= HIGH_TCA
        )
        or
        (
            distance <= HIGH_DISTANCE
            and approach_speed >= HIGH_APPROACH_SPEED
            and convergence >= MIN_CONVERGENCE
        )
        or
        (
            predicted_distance <= MEDIUM_PREDICTED_DISTANCE
            and approach_speed >= HIGH_APPROACH_SPEED
            and convergence >= STRONG_CONVERGENCE
        )
    )

    if high_trajectory and score >= 50:
        return "HIGH", score

    # --------------------------------------------------------
    # MEDIUM trajectory conditions
    # --------------------------------------------------------

    medium_trajectory = (
        (
            distance <= MEDIUM_DISTANCE
            and approach_speed >= MIN_APPROACH_SPEED
        )
        or
        (
            predicted_distance <= MEDIUM_PREDICTED_DISTANCE
            and tca <= MEDIUM_TCA
        )
        or
        (
            tca <= MEDIUM_TCA
            and convergence >= STRONG_CONVERGENCE
        )
    )

    if medium_trajectory and score >= 25:
        return "MEDIUM", score

    return "LOW", 0


# ============================================================
# STRONG OBSERVATION
# ============================================================

def is_strong_observation(metrics, risk):
    """
    Determine whether one observation contains strong evidence
    of a genuine dangerous trajectory.
    """

    if risk == "CRITICAL":
        return True

    if risk != "HIGH":
        return False

    predicted_distance = metrics["predicted_distance"]
    tca = metrics["tca"]
    approach_speed = metrics["approach_speed"]
    distance = metrics["distance"]
    convergence = metrics["convergence"]

    condition_1 = (
        predicted_distance <= STRONG_PREDICTED_DISTANCE
        and tca <= STRONG_TCA_LIMIT
        and approach_speed >= STRONG_APPROACH_SPEED
    )

    condition_2 = (
        distance <= HIGH_DISTANCE
        and approach_speed >= STRONG_APPROACH_SPEED
        and convergence >= STRONG_CONVERGENCE
    )

    return condition_1 or condition_2


# ============================================================
# EVENT MANAGER
# ============================================================

class RiskEventManager:
    """
    Groups frame-level observations into temporal risk events.

    V5.2 improvements:
        - LOW observations do not automatically mean the object
          pair disappeared.
        - Strong observations are tracked by frame number.
        - Strong evidence must be temporally distributed.
        - Only confirmed events are returned.
    """

    def __init__(
        self,
        fps,
        min_event_frames=MIN_EVENT_FRAMES,
        gap_tolerance=EVENT_GAP_TOLERANCE,
    ):
        self.fps = fps

        self.min_event_frames = min_event_frames
        self.gap_tolerance = gap_tolerance

        self.active_events = {}
        self.completed_events = []

    # --------------------------------------------------------
    # Create
    # --------------------------------------------------------

    def _create_event(
        self,
        frame_number,
        pair,
        risk,
        score,
        metrics,
    ):
        strong = is_strong_observation(
            metrics,
            risk,
        )

        event = {
            "pair": pair,

            "start_frame": frame_number,
            "last_frame": frame_number,

            "frames_seen": 1,

            "strong_observations": 1 if strong else 0,
            "strong_frames": (
                [frame_number]
                if strong
                else []
            ),

            # V5.2 event segmentation:
            # consecutive LOW-risk observations.
            "safe_frames": 0,

            "peak_risk": risk,
            "peak_score": score,

            "min_distance": metrics["distance"],
            "min_predicted_distance": (
                metrics["predicted_distance"]
            ),

            "max_approach_speed": (
                metrics["approach_speed"]
            ),

            "min_tca": metrics["tca"],

            "max_convergence": (
                metrics["convergence"]
            ),
        }

        return event

    # --------------------------------------------------------
    # Update metrics
    # --------------------------------------------------------

    def _update_event(
        self,
        event,
        frame_number,
        risk,
        score,
        metrics,
    ):
        event["last_frame"] = frame_number
        event["frames_seen"] += 1

        event["min_distance"] = min(
            event["min_distance"],
            metrics["distance"],
        )

        event["min_predicted_distance"] = min(
            event["min_predicted_distance"],
            metrics["predicted_distance"],
        )

        event["max_approach_speed"] = max(
            event["max_approach_speed"],
            metrics["approach_speed"],
        )

        event["min_tca"] = min(
            event["min_tca"],
            metrics["tca"],
        )

        event["max_convergence"] = max(
            event["max_convergence"],
            metrics["convergence"],
        )

        risk_priority = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
            "CRITICAL": 3,
        }

        if risk_priority.get(risk, 0) > risk_priority.get(
            event["peak_risk"],
            0,
        ):
            event["peak_risk"] = risk

        event["peak_score"] = max(
            event["peak_score"],
            score,
        )

        # ----------------------------------------------------
        # Strong evidence
        # ----------------------------------------------------

        if is_strong_observation(metrics, risk):

            strong_frames = event["strong_frames"]

            if not strong_frames:
                strong_frames.append(frame_number)
                event["strong_observations"] += 1

            else:
                last_strong_frame = strong_frames[-1]

                if (
                    frame_number - last_strong_frame
                    >= MIN_STRONG_FRAME_GAP
                ):
                    strong_frames.append(frame_number)
                    event["strong_observations"] += 1

    # --------------------------------------------------------
    # Observe
    # --------------------------------------------------------

    def observe(
        self,
        frame_number,
        pair,
        risk,
        score,
        metrics,
    ):
        """
        Process one valid pair observation.

        V5.2:
            LOW-risk observations keep the pair alive temporarily,
            but a sustained LOW period ends the current risk episode.
        """

        event = self.active_events.get(pair)

        # ----------------------------------------------------
        # New pair
        # ----------------------------------------------------

        if event is None:

            if risk == "LOW":
                return

            self.active_events[pair] = self._create_event(
                frame_number,
                pair,
                risk,
                score,
                metrics,
            )

            return

        # ----------------------------------------------------
        # Pair continued
        # ----------------------------------------------------

        frame_gap = frame_number - event["last_frame"]

        if frame_gap <= self.gap_tolerance + 1:

            # ------------------------------------------------
            # V5.2 SAFE GAP
            # ------------------------------------------------
            #
            # LOW risk means the pair is currently not in a
            # meaningful risk state. We allow a short LOW
            # period so that tiny fluctuations don't fragment
            # the event immediately.
            #
            # If LOW continues long enough, the event ends.
            # ------------------------------------------------

            if risk == "LOW":

                event["safe_frames"] += max(1, frame_gap)
                event["last_frame"] = frame_number

                if (
                    event["safe_frames"]
                    >= EVENT_SAFE_GAP_FRAMES
                ):
                    self._finalize_event(pair)

                return

            # ------------------------------------------------
            # Risk returned.
            # ------------------------------------------------

            event["safe_frames"] = 0

            self._update_event(
                event,
                frame_number,
                risk,
                score,
                metrics,
            )

            return

        # ----------------------------------------------------
        # Previous event ended because the observation gap
        # exceeded the detection-gap tolerance.
        # ----------------------------------------------------

        self._finalize_event(pair)

        if risk != "LOW":

            self.active_events[pair] = self._create_event(
                frame_number,
                pair,
                risk,
                score,
                metrics,
            )

    # --------------------------------------------------------
    # Missing pair
    # --------------------------------------------------------

    def mark_missing_pairs(
        self,
        frame_number,
        observed_pairs,
    ):
        """
        Finalize pairs that have genuinely disappeared from
        the current frame stream.

        This is intentionally separate from LOW risk.
        """

        to_finalize = []

        for pair, event in self.active_events.items():

            if pair in observed_pairs:
                continue

            if (
                frame_number - event["last_frame"]
                > self.gap_tolerance
            ):
                to_finalize.append(pair)

        for pair in to_finalize:
            self._finalize_event(pair)

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    def _is_confirmed(self, event):
        if event["frames_seen"] < self.min_event_frames:
            return False

        strong_frames = event["strong_frames"]

        if len(strong_frames) < MIN_STRONG_OBSERVATIONS:
            return False

        strong_span = (
            strong_frames[-1] - strong_frames[0]
        )

        if strong_span < MIN_STRONG_SPAN_FRAMES:
            return False

        return True

    # --------------------------------------------------------
    # Finalize
    # --------------------------------------------------------

    def _finalize_event(self, pair):
        event = self.active_events.pop(
            pair,
            None,
        )

        if event is None:
            return

        if not self._is_confirmed(event):
            return

        event["duration_seconds"] = (
            event["last_frame"]
            - event["start_frame"]
        ) / self.fps

        event["confirmed"] = True

        self.completed_events.append(event)

    # --------------------------------------------------------
    # Finalize everything
    # --------------------------------------------------------

    def finalize_all(self):
        for pair in list(self.active_events.keys()):
            self._finalize_event(pair)


# ============================================================
# FORMAT EVENT
# ============================================================

def print_event(event, index):
    pair = event["pair"]

    min_tca = event["min_tca"]

    tca_text = (
        f"{min_tca:.2f}"
        if np.isfinite(min_tca)
        else "N/A"
    )

    print()
    print("=" * 70)
    print(f"Event #{index}")

    print(
        f"IDs: {pair[0]} <-> {pair[1]}"
    )

    print(
        f"Frames: "
        f"{event['start_frame']} - "
        f"{event['last_frame']}"
    )

    print(
        f"Duration: "
        f"{event['duration_seconds']:.2f} sec"
    )

    print(
        f"Frames Observed: "
        f"{event['frames_seen']}"
    )

    print(
        f"Strong Observations: "
        f"{event['strong_observations']}"
    )

    print(
        f"Strong Frames: "
        f"{event['strong_frames']}"
    )

    print(
        f"Peak Risk: "
        f"{event['peak_risk']}"
    )

    print(
        f"Peak Score: "
        f"{event['peak_score']}"
    )

    print(
        f"Minimum Distance: "
        f"{event['min_distance']:.2f} px"
    )

    print(
        f"Minimum Predicted Distance: "
        f"{event['min_predicted_distance']:.2f} px"
    )

    print(
        f"Maximum Approach Speed: "
        f"{event['max_approach_speed']:.2f} px/sec"
    )

    print(
        f"Minimum TCA: "
        f"{tca_text} sec"
    )

    print(
        f"Maximum Convergence: "
        f"{event['max_convergence']:.2f}"
    )

    print("Confirmed: True")


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Could not open the video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    print(f"Video FPS: {fps}")

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    previous_positions = {}

    velocity_history = defaultdict(
        lambda: deque(
            maxlen=HISTORY_SIZE
        )
    )

    event_manager = RiskEventManager(
        fps=fps,
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    risk_counts = {
        "MEDIUM": 0,
        "HIGH": 0,
        "CRITICAL": 0,
    }

    frame_number = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            verbose=False,
        )

        if not results:
            event_manager.mark_missing_pairs(
                frame_number,
                set(),
            )
            continue

        result = results[0]

        observed_pairs = set()

        current_positions = {}

        # ----------------------------------------------------
        # Extract tracked objects
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            if boxes.id is not None:

                track_ids = (
                    boxes.id.int()
                    .cpu()
                    .tolist()
                )

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .numpy()
                )

                class_ids = (
                    boxes.cls.int()
                    .cpu()
                    .tolist()
                )

                for track_id, box, class_id in zip(
                    track_ids,
                    xyxy,
                    class_ids,
                ):

                    position = bottom_center(box)

                    current_positions[track_id] = position

                    # ------------------------------------------------
                    # Velocity
                    # ------------------------------------------------

                    if track_id in previous_positions:

                        displacement = (
                            position
                            - previous_positions[track_id]
                        )

                        velocity = (
                            displacement * fps
                        )

                        velocity_magnitude = float(
                            np.linalg.norm(velocity)
                        )

                        if (
                            velocity_magnitude
                            <= MAX_REASONABLE_VELOCITY
                        ):
                            velocity_history[
                                track_id
                            ].append(velocity)

                    previous_positions[
                        track_id
                    ] = position

        # --------------------------------------------------------
        # Pairwise trajectory analysis
        # --------------------------------------------------------

        track_ids = list(
            current_positions.keys()
        )

        for i in range(len(track_ids)):

            id_1 = track_ids[i]

            if not velocity_history[id_1]:
                continue

            velocity_1 = smooth_velocity(
                velocity_history[id_1]
            )

            for j in range(i + 1, len(track_ids)):

                id_2 = track_ids[j]

                if not velocity_history[id_2]:
                    continue

                velocity_2 = smooth_velocity(
                    velocity_history[id_2]
                )

                # ----------------------------------------------------
                # Relative velocity sanity check
                # ----------------------------------------------------

                relative_velocity = (
                    velocity_2
                    - velocity_1
                )

                relative_velocity_magnitude = float(
                    np.linalg.norm(relative_velocity)
                )

                if (
                    relative_velocity_magnitude
                    > MAX_REASONABLE_RELATIVE_VELOCITY
                ):
                    continue

                # ----------------------------------------------------
                # Trajectory metrics
                # ----------------------------------------------------

                metrics = calculate_trajectory_metrics(
                    current_positions[id_1],
                    velocity_1,
                    current_positions[id_2],
                    velocity_2,
                )

                risk, score = calculate_risk(
                    metrics
                )

                pair = tuple(
                    sorted(
                        (id_1, id_2)
                    )
                )

                observed_pairs.add(pair)

                event_manager.observe(
                    frame_number,
                    pair,
                    risk,
                    score,
                    metrics,
                )

                # ----------------------------------------------------
                # Risk output
                # ----------------------------------------------------

                if risk != "LOW":

                    risk_counts[risk] += 1

                    tca = metrics["tca"]

                    tca_text = (
                        f"{tca:.2f}"
                        if np.isfinite(tca)
                        else "N/A"
                    )

                    print(
                        f"Frame: {frame_number} | "
                        f"ID: {id_1} <-> ID: {id_2} | "
                        f"{risk} | "
                        f"Score: {score} | "
                        f"Distance: "
                        f"{metrics['distance']:.2f}px | "
                        f"Predicted: "
                        f"{metrics['predicted_distance']:.2f}px | "
                        f"Approach: "
                        f"{metrics['approach_speed']:.2f}px/s | "
                        f"TCA: "
                        f"{tca_text}s | "
                        f"Convergence: "
                        f"{metrics['convergence']:.2f}"
                    )

        # --------------------------------------------------------
        # Detect genuinely missing pairs
        # --------------------------------------------------------

        event_manager.mark_missing_pairs(
            frame_number,
            observed_pairs,
        )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    video.release()

    event_manager.finalize_all()

    # ========================================================
    # V5.3 MACHINE-READABLE EVENT EXPORT
    # ========================================================


    event_output_path = "output/confirmed_risk_events_v53.json"

    Path("output").mkdir(
        parents=True,
        exist_ok=True,
    )

    events = event_manager.completed_events

    event_export = {
        "version": "V5.3",
        "fps": fps,
        "video_path": VIDEO_PATH,
        "confirmed_event_count": len(events),
        "events": events,
    }

    with open(
        event_output_path,
        "w",
        encoding="utf-8",
    ) as event_file:

        json.dump(
            event_export,
            event_file,
            indent=2,
        )

    print(
        f"Confirmed events exported to: "
        f"{event_output_path}"
    )

    print()
    print("=" * 70)
    print("V5.2 RISK ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"MEDIUM observations: "
        f"{risk_counts['MEDIUM']}"
    )

    print(
        f"HIGH observations: "
        f"{risk_counts['HIGH']}"
    )

    print(
        f"CRITICAL observations: "
        f"{risk_counts['CRITICAL']}"
    )

    total = sum(risk_counts.values())

    print(
        f"TOTAL risk observations: "
        f"{total}"
    )

    print()
    print(
        f"Confirmed risk events: "
        f"{len(event_manager.completed_events)}"
    )

    for index, event in enumerate(
        event_manager.completed_events,
        start=1,
    ):
        print_event(
            event,
            index,
        )


if __name__ == "__main__":
    main()
