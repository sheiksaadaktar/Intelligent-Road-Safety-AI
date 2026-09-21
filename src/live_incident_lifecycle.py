import json
from pathlib import Path


OUTPUT_PATH = Path("output/live_incidents.json")


class LiveIncidentLifecycle:
    """
    Tracks the lifecycle of temporally confirmed live risk events.

    Lifecycle:
        CONFIRMED -> ACTIVE -> RESOLVED

    The event identity is:
        (sorted object pair, event start frame)
    """

    VALID_STATES = {
        "CONFIRMED",
        "ACTIVE",
        "RESOLVED",
    }

    def __init__(
        self,
        fps,
        output_path=OUTPUT_PATH,
    ):
        self.fps = fps
        self.output_path = Path(output_path)

        self.incidents = []
        self.incidents_by_key = {}

        self.next_incident_number = 1

    # ========================================================
    # EVENT IDENTITY
    # ========================================================

    @staticmethod
    def event_key(event):
        pair = tuple(
            sorted(
                event["pair"]
            )
        )

        return (
            pair,
            event["start_frame"],
        )

    # ========================================================
    # ALERT LOOKUP
    # ========================================================

    @staticmethod
    def alert_key(alert):
        pair = tuple(
            sorted(
                alert["pair"]
            )
        )

        return (
            pair,
            alert["event"]["start_frame"],
        )

    # ========================================================
    # HELPERS
    # ========================================================

    def _next_incident_id(self):

        incident_id = (
            f"LIVE-INC-{self.next_incident_number:04d}"
        )

        self.next_incident_number += 1

        return incident_id

    def _duration_seconds(
        self,
        start_frame,
        last_frame,
    ):

        if self.fps <= 0:
            return 0.0

        return (
            last_frame - start_frame
        ) / self.fps

    @staticmethod
    def _safe_float(value):

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    # ========================================================
    # INCIDENT CREATION
    # ========================================================

    def _create_incident(
        self,
        event,
        alert,
        frame_number,
    ):

        key = self.event_key(
            event
        )

        incident_id = (
            self._next_incident_id()
        )

        alert_level = (
            alert["alert_level"]
            if alert is not None
            else "MONITOR"
        )

        priority = (
            alert["priority"]
            if alert is not None
            else 3
        )

        state = "CONFIRMED"

        incident = {
            "incident_id": incident_id,

            "objects": {
                "pair": list(
                    event["pair"]
                ),
            },

            "classification": {
                "risk": event[
                    "peak_risk"
                ],
                "score": event[
                    "peak_score"
                ],
                "alert_level": alert_level,
                "priority": priority,
            },

            "lifecycle": {
                "state": state,
                "confirmed_frame": frame_number,
                "active_frame": None,
                "resolved_frame": None,
            },

            "event_timing": {
                "start_frame": event[
                    "start_frame"
                ],
                "last_observed_frame": event[
                    "last_frame"
                ],
                "duration_seconds": self._duration_seconds(
                    event["start_frame"],
                    event["last_frame"],
                ),
            },

            "trajectory_evidence": {
                "min_distance_pixels": self._safe_float(
                    event["min_distance"]
                ),
                "min_predicted_distance_pixels": self._safe_float(
                    event[
                        "min_predicted_distance"
                    ]
                ),
                "min_tca_seconds": self._safe_float(
                    event["min_tca"]
                ),
                "max_approach_speed_pixels_per_second": self._safe_float(
                    event["max_approach_speed"]
                ),
                "max_convergence": self._safe_float(
                    event["max_convergence"]
                ),
            },

            "confirmation_evidence": {
                "frames_seen": event[
                    "frames_seen"
                ],
                "strong_observations": event[
                    "strong_observations"
                ],
                "strong_frames": list(
                    event["strong_frames"]
                ),
                "confirmed": bool(
                    event.get(
                        "confirmed",
                        True,
                    )
                ),
            },

            "source_alert": {
                "alert_id": (
                    alert["alert_id"]
                    if alert is not None
                    else None
                ),
            },

            "resolution": {
                "resolved": False,
                "reason": None,
            },
        }

        self.incidents.append(
            incident
        )

        self.incidents_by_key[
            key
        ] = incident

        return incident

    # ========================================================
    # INCIDENT UPDATE
    # ========================================================

    def _update_from_event(
        self,
        incident,
        event,
    ):

        lifecycle = incident[
            "lifecycle"
        ]

        if lifecycle["state"] == "CONFIRMED":

            lifecycle["state"] = "ACTIVE"

            if lifecycle[
                "active_frame"
            ] is None:

                lifecycle[
                    "active_frame"
                ] = event[
                    "last_frame"
                ]

        elif lifecycle["state"] == "ACTIVE":

            lifecycle[
                "active_frame"
            ] = event[
                "last_frame"
            ]

        incident[
            "event_timing"
        ][
            "last_observed_frame"
        ] = event[
            "last_frame"
        ]

        incident[
            "event_timing"
        ][
            "duration_seconds"
        ] = self._duration_seconds(
            event["start_frame"],
            event["last_frame"],
        )

        # Keep the strongest trajectory evidence
        # synchronized with the live event.
        evidence = incident[
            "trajectory_evidence"
        ]

        evidence[
            "min_distance_pixels"
        ] = min(
            evidence[
                "min_distance_pixels"
            ],
            self._safe_float(
                event["min_distance"]
            ),
        )

        evidence[
            "min_predicted_distance_pixels"
        ] = min(
            evidence[
                "min_predicted_distance_pixels"
            ],
            self._safe_float(
                event[
                    "min_predicted_distance"
                ]
            ),
        )

        evidence[
            "min_tca_seconds"
        ] = min(
            evidence[
                "min_tca_seconds"
            ],
            self._safe_float(
                event["min_tca"]
            ),
        )

        evidence[
            "max_approach_speed_pixels_per_second"
        ] = max(
            evidence[
                "max_approach_speed_pixels_per_second"
            ],
            self._safe_float(
                event[
                    "max_approach_speed"
                ]
            ),
        )

        evidence[
            "max_convergence"
        ] = max(
            evidence[
                "max_convergence"
            ],
            self._safe_float(
                event[
                    "max_convergence"
                ]
            ),
        )

        incident[
            "confirmation_evidence"
        ][
            "frames_seen"
        ] = event[
            "frames_seen"
        ]

        incident[
            "confirmation_evidence"
        ][
            "strong_observations"
        ] = event[
            "strong_observations"
        ]

        incident[
            "confirmation_evidence"
        ][
            "strong_frames"
        ] = list(
            event[
                "strong_frames"
            ]
        )

        # Keep the highest severity reached.
        risk_order = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
            "CRITICAL": 3,
        }

        current_risk = incident[
            "classification"
        ][
            "risk"
        ]

        event_risk = event[
            "peak_risk"
        ]

        if (
            risk_order.get(
                event_risk,
                0,
            )
            > risk_order.get(
                current_risk,
                0,
            )
        ):

            incident[
                "classification"
            ][
                "risk"
            ] = event_risk

        incident[
            "classification"
        ][
            "score"
        ] = max(
            incident[
                "classification"
            ][
                "score"
            ],
            event[
                "peak_score"
            ],
        )

    # ========================================================
    # CONFIRMED ALERT PROCESSING
    # ========================================================

    def process_alerts(
        self,
        alerts,
        active_events,
        frame_number,
    ):
        """
        Synchronize lifecycle state with the current live
        alert pipeline.

        Newly confirmed alerts create CONFIRMED incidents.

        Existing active events transition:
            CONFIRMED -> ACTIVE
        """

        active_by_key = {
            self.event_key(event): event
            for event in active_events.values()
        }

        alert_by_key = {
            self.alert_key(alert): alert
            for alert in alerts
        }

        for key, event in active_by_key.items():

            if key not in alert_by_key:
                continue

            alert = alert_by_key[key]

            if key not in self.incidents_by_key:

                self._create_incident(
                    event=event,
                    alert=alert,
                    frame_number=frame_number,
                )

            else:

                incident = (
                    self.incidents_by_key[key]
                )

                if incident[
                    "lifecycle"
                ][
                    "state"
                ] != "RESOLVED":

                    self._update_from_event(
                        incident,
                        event,
                    )

        return self.active_incidents()

    # ========================================================
    # RESOLUTION
    # ========================================================

    def resolve_completed_events(
        self,
        completed_events,
        frame_number,
    ):
        """
        Mark incidents RESOLVED when RiskEventManager has
        finalized their corresponding events.
        """

        resolved = []

        for event in completed_events:

            key = self.event_key(
                event
            )

            incident = (
                self.incidents_by_key.get(
                    key
                )
            )

            if incident is None:
                continue

            lifecycle = incident[
                "lifecycle"
            ]

            if lifecycle[
                "state"
            ] == "RESOLVED":

                continue

            # A confirmed event that was finalized is resolved.
            lifecycle[
                "state"
            ] = "RESOLVED"

            lifecycle[
                "resolved_frame"
            ] = frame_number

            incident[
                "event_timing"
            ][
                "last_observed_frame"
            ] = event[
                "last_frame"
            ]

            incident[
                "event_timing"
            ][
                "duration_seconds"
            ] = self._duration_seconds(
                event["start_frame"],
                event["last_frame"],
            )

            incident[
                "resolution"
            ][
                "resolved"
            ] = True

            incident[
                "resolution"
            ][
                "reason"
            ] = "Risk event finalized"

            # Final evidence synchronization.
            self._update_from_event(
                incident,
                event,
            )

            # _update_from_event moves RESOLVED events
            # back to ACTIVE if called after resolution.
            lifecycle[
                "state"
            ] = "RESOLVED"

            lifecycle[
                "resolved_frame"
            ] = frame_number

            resolved.append(
                incident
            )

        return resolved

    # ========================================================
    # FINAL RESOLUTION
    # ========================================================

    def resolve_all(
        self,
        frame_number,
    ):

        resolved = []

        for incident in self.incidents:

            lifecycle = incident[
                "lifecycle"
            ]

            if lifecycle[
                "state"
            ] == "RESOLVED":

                continue

            lifecycle[
                "state"
            ] = "RESOLVED"

            lifecycle[
                "resolved_frame"
            ] = frame_number

            incident[
                "resolution"
            ][
                "resolved"
            ] = True

            incident[
                "resolution"
            ][
                "reason"
            ] = "Monitoring session ended"

            resolved.append(
                incident
            )

        return resolved

    # ========================================================
    # STATE QUERIES
    # ========================================================

    def active_incidents(self):

        return [
            incident
            for incident in self.incidents
            if incident[
                "lifecycle"
            ][
                "state"
            ] in (
                "CONFIRMED",
                "ACTIVE",
            )
        ]

    def confirmed_incidents(self):

        return [
            incident
            for incident in self.incidents
            if incident[
                "lifecycle"
            ][
                "state"
            ] == "CONFIRMED"
        ]

    def resolved_incidents(self):

        return [
            incident
            for incident in self.incidents
            if incident[
                "lifecycle"
            ][
                "state"
            ] == "RESOLVED"
        ]

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(self):

        total = len(
            self.incidents
        )

        state_counts = {
            "CONFIRMED": 0,
            "ACTIVE": 0,
            "RESOLVED": 0,
        }

        risk_counts = {}

        alert_counts = {}

        object_involvement = {}

        durations = []

        for incident in self.incidents:

            state = incident[
                "lifecycle"
            ][
                "state"
            ]

            state_counts[
                state
            ] = (
                state_counts.get(
                    state,
                    0,
                )
                + 1
            )

            risk = incident[
                "classification"
            ][
                "risk"
            ]

            risk_counts[
                risk
            ] = (
                risk_counts.get(
                    risk,
                    0,
                )
                + 1
            )

            alert_level = incident[
                "classification"
            ][
                "alert_level"
            ]

            alert_counts[
                alert_level
            ] = (
                alert_counts.get(
                    alert_level,
                    0,
                )
                + 1
            )

            pair = incident[
                "objects"
            ][
                "pair"
            ]

            for object_id in pair:

                object_key = str(
                    object_id
                )

                object_involvement[
                    object_key
                ] = (
                    object_involvement.get(
                        object_key,
                        0,
                    )
                    + 1
                )

            durations.append(
                incident[
                    "event_timing"
                ][
                    "duration_seconds"
                ]
            )

        average_duration = (
            sum(durations) / len(durations)
            if durations
            else 0.0
        )

        return {
            "total_incidents": total,

            "state_counts": state_counts,

            "risk_counts": risk_counts,

            "alert_level_counts": alert_counts,

            "object_involvement": object_involvement,

            "duration_seconds": {
                "average": round(
                    average_duration,
                    4,
                ),
                "maximum": round(
                    max(durations),
                    4,
                ) if durations else 0.0,
                "minimum": round(
                    min(durations),
                    4,
                ) if durations else 0.0,
            },
        }

    # ========================================================
    # OUTPUT
    # ========================================================

    def build_output(
        self,
        source_video=None,
    ):

        return {
            "version": "Live-Incident-2.0",

            "source": {
                "video": source_video,
                "fps": self.fps,
            },

            "lifecycle_policy": {
                "confirmed": (
                    "Temporal confirmation threshold reached"
                ),
                "active": (
                    "Confirmed event remains active"
                ),
                "resolved": (
                    "Risk event finalized by temporal engine"
                ),
            },

            "statistics": self.statistics(),

            "incidents": self.incidents,
        }

    def save(
        self,
        source_video=None,
    ):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = self.build_output(
            source_video=source_video,
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
            )

        return self.output_path