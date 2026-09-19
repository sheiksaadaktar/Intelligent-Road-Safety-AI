import json
from pathlib import Path

from risk_analysis import RiskEventManager


OUTPUT_PATH = Path("output/live_alerts.json")


ALERT_DEFINITIONS = {
    "CRITICAL": {
        "priority": 1,
        "state": "ACTIVE",
        "message": "Immediate collision-risk alert",
    },
    "HIGH": {
        "priority": 2,
        "state": "ACTIVE",
        "message": "Priority collision-risk alert",
    },
    "MEDIUM": {
        "priority": 3,
        "state": "MONITORING",
        "message": "Monitor developing road-safety risk",
    },
}


class LiveAlertPipeline:
    """
    Connects trajectory risk observations to temporally confirmed live alerts.

    Important:
    - Active events are checked for confirmation during processing.
    - Completed events are also checked during finalization.
    - Each confirmed event generates exactly one alert.
    """

    def __init__(self, fps, output_path=OUTPUT_PATH):
        self.fps = fps
        self.output_path = Path(output_path)

        self.event_manager = RiskEventManager(fps=fps)

        self.alerted_events = set()
        self.alerts = []
        self.next_alert_number = 1

    def _get_definition(self, risk):
        return ALERT_DEFINITIONS.get(
            risk,
            {
                "priority": 3,
                "state": "MONITORING",
                "message": "Monitor developing road-safety risk",
            },
        )

    @staticmethod
    def _event_key(event):
        pair = tuple(sorted(event["pair"]))
        return pair, event["start_frame"]

    def _build_alert(self, event):
        peak_risk = event["peak_risk"]
        definition = self._get_definition(peak_risk)

        alert_id = f"LIVE-{self.next_alert_number:04d}"
        self.next_alert_number += 1

        return {
            "alert_id": alert_id,
            "alert_index": len(self.alerts) + 1,
            "pair": list(event["pair"]),
            "alert_level": (
                "IMMEDIATE_ALERT"
                if peak_risk == "CRITICAL"
                else "PRIORITY_ALERT"
                if peak_risk == "HIGH"
                else "MONITOR"
            ),
            "priority": definition["priority"],
            "state": definition["state"],
            "message": definition["message"],
            "risk": {
                "peak_risk": event["peak_risk"],
                "peak_score": event["peak_score"],
            },
            "event": {
                "start_frame": event["start_frame"],
                "last_frame": event["last_frame"],
                "duration_seconds": event.get(
                    "duration_seconds",
                    (event["last_frame"] - event["start_frame"]) / self.fps
                    if self.fps > 0
                    else 0.0,
                ),
            },
            "trajectory_evidence": {
                "min_distance_pixels": event["min_distance"],
                "min_predicted_distance_pixels": event[
                    "min_predicted_distance"
                ],
                "min_tca_seconds": event["min_tca"],
                "max_approach_speed_pixels_per_second": event[
                    "max_approach_speed"
                ],
                "max_convergence": event["max_convergence"],
            },
            "confirmation": {
                "frames_seen": event["frames_seen"],
                "strong_observations": event["strong_observations"],
                "strong_frames": event["strong_frames"],
                "confirmed": event.get("confirmed", True),
            },
        }

    def _add_confirmed_event(self, event):
        key = self._event_key(event)

        if key in self.alerted_events:
            return False

        alert = self._build_alert(event)

        self.alerts.append(alert)
        self.alerted_events.add(key)

        return True

    def _collect_live_confirmed_alerts(self):
        """
        Check active events immediately after each observation.

        This reuses RiskEventManager's established confirmation logic
        instead of duplicating the confirmation thresholds here.
        """

        generated = 0

        active_events = getattr(self.event_manager, "active_events", {})

        for event in active_events.values():

            if self._event_key(event) in self.alerted_events:
                continue

            if self.event_manager._is_confirmed(event):
                event["confirmed"] = True

                if self._add_confirmed_event(event):
                    generated += 1

        return generated

    def _collect_completed_alerts(self):
        """
        Collect confirmed events that were finalized because the event
        ended or the stream reached its final frame.
        """

        generated = 0

        for event in self.event_manager.completed_events:

            if not event.get("confirmed", False):
                continue

            if self._add_confirmed_event(event):
                generated += 1

        return generated

    def observe(self, frame_number, pair, risk, score, metrics):
        """
        Feed one trajectory-risk observation into the temporal
        confirmation engine.

        Returns True when this observation causes a new live alert.
        """

        self.event_manager.observe(
            frame_number=frame_number,
            pair=pair,
            risk=risk,
            score=score,
            metrics=metrics,
        )

        return self._collect_live_confirmed_alerts() > 0

    def mark_missing_pairs(self, frame_number, observed_pairs):
        """
        Inform the event manager about pairs no longer observed.
        """

        normalized_pairs = {
            tuple(sorted(pair))
            for pair in observed_pairs
        }

        self.event_manager.mark_missing_pairs(
            frame_number,
            normalized_pairs,
        )

        return self._collect_completed_alerts()

    def finalize(self):
        """
        Finalize all active events and collect any confirmed alerts.
        """

        self.event_manager.finalize_all()

        return self._collect_completed_alerts()

    def statistics(self):
        total = len(self.alerts)

        immediate = sum(
            1
            for alert in self.alerts
            if alert["alert_level"] == "IMMEDIATE_ALERT"
        )

        priority = sum(
            1
            for alert in self.alerts
            if alert["alert_level"] == "PRIORITY_ALERT"
        )

        monitor = sum(
            1
            for alert in self.alerts
            if alert["alert_level"] == "MONITOR"
        )

        risk_counts = {}

        for alert in self.alerts:
            risk = alert["risk"]["peak_risk"]
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        state_counts = {}

        for alert in self.alerts:
            state = alert["state"]
            state_counts[state] = state_counts.get(state, 0) + 1

        def percentage(value):
            if total == 0:
                return 0.0
            return round((value / total) * 100, 2)

        return {
            "total_alerts": total,
            "alert_counts": {
                "IMMEDIATE_ALERT": immediate,
                "PRIORITY_ALERT": priority,
                "MONITOR": monitor,
            },
            "alert_percentages": {
                "IMMEDIATE_ALERT": percentage(immediate),
                "PRIORITY_ALERT": percentage(priority),
                "MONITOR": percentage(monitor),
            },
            "risk_counts": risk_counts,
            "state_counts": state_counts,
        }

    def build_output(self, source_video=None):
        return {
            "version": "Live-Alert-1.0",
            "source": {
                "video": source_video,
                "fps": self.fps,
            },
            "confirmation_policy": {
                "min_event_frames": self.event_manager.min_event_frames,
                "gap_tolerance": self.event_manager.gap_tolerance,
                "min_strong_observations": 3,
                "min_strong_span_frames": 4,
                "safe_gap_frames": 15,
            },
            "statistics": self.statistics(),
            "alerts": self.alerts,
        }

    def save(self, source_video=None):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        output = self.build_output(source_video=source_video)

        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

        return self.output_path


def run_live_confirmation_test():
    """
    Controlled timing test.

    The important assertion is that the alert is generated while the
    event is STILL ACTIVE, before finalize() is called.
    """

    fps = 60

    pipeline = LiveAlertPipeline(
        fps=fps,
        output_path=Path("output/live_alert_pipeline_timing_test.json"),
    )

    metrics = {
        "distance": 80.0,
        "relative_speed": 120.0,
        "approach_speed": 100.0,
        "convergence": 0.90,
        "tca": 0.80,
        "predicted_distance": 30.0,
        "converging": True,
    }

    pair = (101, 102)

    test_frames = [100, 102, 106, 110, 112]

    print("")
    print("======================================================================")
    print("INTELLIGENT ROAD SAFETY AI")
    print("LIVE ALERT PIPELINE TIMING TEST")
    print("======================================================================")
    print("")
    print("Feeding observations...")
    print("")

    alert_generated_before_finalization = False

    for frame in test_frames:

        generated = pipeline.observe(
            frame_number=frame,
            pair=pair,
            risk="HIGH",
            score=90,
            metrics=metrics,
        )

        print(
            f"Frame {frame:>3} | "
            f"Alert generated: {generated} | "
            f"Total alerts: {len(pipeline.alerts)}"
        )

        if generated:
            alert_generated_before_finalization = True
            print("")
            print(">>> LIVE ALERT GENERATED WHILE EVENT IS ACTIVE <<<")
            print("")

            break

    if not alert_generated_before_finalization:
        print("")
        print("ERROR: No live alert was generated before finalization.")
        print("")

        pipeline.finalize()

        print(
            f"Alerts after finalization: {len(pipeline.alerts)}"
        )

        raise RuntimeError(
            "Live confirmation timing test failed."
        )

    active_count_before_finalize = len(
        pipeline.event_manager.active_events
    )

    if active_count_before_finalize == 0:
        raise RuntimeError(
            "Alert was generated, but the event was not still active."
        )

    print(
        f"Active events before finalization: "
        f"{active_count_before_finalize}"
    )

    pipeline.finalize()

    if len(pipeline.alerts) != 1:
        raise RuntimeError(
            "Finalization created a duplicate alert."
        )

    output_path = pipeline.save(
        source_video="timing_test"
    )

    print("")
    print("===== LIVE TIMING TEST SUMMARY =====")
    print(f"Alerts before finalization: 1")
    print(f"Alerts after finalization: {len(pipeline.alerts)}")
    print(
        f"Alert ID: {pipeline.alerts[0]['alert_id']}"
    )
    print(
        f"Alert level: {pipeline.alerts[0]['alert_level']}"
    )
    print(
        f"Event still active at alert time: YES"
    )
    print("")
    print(f"Test output: {output_path}")
    print("======================================================================")


if __name__ == "__main__":
    run_live_confirmation_test()
