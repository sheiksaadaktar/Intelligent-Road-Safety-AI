import json
import math
from pathlib import Path


ALERTS_PATH = Path("output/live_alerts.json")
INCIDENTS_PATH = Path("output/live_incidents.json")
ANALYTICS_PATH = Path("output/safety_analytics.json")
OUTPUT_PATH = Path("output/safety_evaluation.json")


EXPECTED_ALERT_LEVELS = {
    "CRITICAL": "IMMEDIATE_ALERT",
    "HIGH": "PRIORITY_ALERT",
    "MEDIUM": "MONITOR",
}


class SafetyEvaluation:
    def __init__(self):
        self.checks = []
        self.failures = 0
        self.warnings = 0

    def check(self, name, passed, details="", warning=False):
        if passed:
            status = "PASS"
        elif warning:
            status = "WARN"
            self.warnings += 1
        else:
            status = "FAIL"
            self.failures += 1

        self.checks.append(
            {
                "name": name,
                "status": status,
                "details": details,
            }
        )

    @staticmethod
    def load_json(path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def pair_from_alert(alert):
        return tuple(
            sorted(
                alert["pair"]
            )
        )

    @staticmethod
    def pair_from_incident(incident):
        return tuple(
            sorted(
                incident["objects"]["pair"]
            )
        )

    @staticmethod
    def finite_number(value):
        return (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    def evaluate_alert_structure(self, alerts):
        alert_ids = [
            alert.get("alert_id")
            for alert in alerts
        ]

        self.check(
            "Unique alert IDs",
            len(alert_ids)
            == len(set(alert_ids)),
            (
                f"{len(alert_ids)} alerts inspected"
            ),
        )

        for alert in alerts:
            alert_id = alert.get(
                "alert_id",
                "UNKNOWN",
            )

            risk = alert.get(
                "risk",
                {},
            ).get(
                "peak_risk"
            )

            score = alert.get(
                "risk",
                {},
            ).get(
                "peak_score"
            )

            alert_level = alert.get(
                "alert_level"
            )

            expected_level = (
                EXPECTED_ALERT_LEVELS.get(
                    risk
                )
            )

            self.check(
                f"{alert_id}: alert-level mapping",
                expected_level == alert_level,
                (
                    f"Risk={risk}, "
                    f"expected={expected_level}, "
                    f"actual={alert_level}"
                ),
            )

            confirmation = alert.get(
                "confirmation",
                {},
            )

            frames_seen = confirmation.get(
                "frames_seen",
                0,
            )

            strong_observations = confirmation.get(
                "strong_observations",
                0,
            )

            confirmed = confirmation.get(
                "confirmed",
                False,
            )

            self.check(
                f"{alert_id}: confirmation evidence",
                (
                    confirmed
                    and frames_seen >= 5
                    and strong_observations >= 3
                ),
                (
                    f"frames_seen={frames_seen}, "
                    f"strong_observations="
                    f"{strong_observations}, "
                    f"confirmed={confirmed}"
                ),
            )

            trajectory = alert.get(
                "trajectory_evidence",
                {},
            )

            trajectory_values = {
                "minimum_distance_pixels":
                    trajectory.get(
                        "min_distance_pixels"
                    ),
                "minimum_predicted_distance_pixels":
                    trajectory.get(
                        "min_predicted_distance_pixels"
                    ),
                "minimum_tca_seconds":
                    trajectory.get(
                        "min_tca_seconds"
                    ),
                "maximum_approach_speed":
                    trajectory.get(
                        "max_approach_speed_pixels_per_second"
                    ),
                "maximum_convergence":
                    trajectory.get(
                        "max_convergence"
                    ),
            }

            valid_trajectory = all(
                self.finite_number(value)
                for value in trajectory_values.values()
            )

            self.check(
                f"{alert_id}: trajectory values",
                valid_trajectory,
                (
                    "All trajectory evidence "
                    "values must be finite numbers"
                ),
            )

            if valid_trajectory:
                self.check(
                    f"{alert_id}: non-negative distance",
                    (
                        trajectory_values[
                            "minimum_distance_pixels"
                        ] >= 0
                        and
                        trajectory_values[
                            "minimum_predicted_distance_pixels"
                        ] >= 0
                    ),
                    "Distance values are non-negative",
                )

                self.check(
                    f"{alert_id}: non-negative TCA",
                    (
                        trajectory_values[
                            "minimum_tca_seconds"
                        ] >= 0
                    ),
                    "TCA is non-negative",
                )

                self.check(
                    f"{alert_id}: convergence range",
                    (
                        -1.0
                        <= trajectory_values[
                            "maximum_convergence"
                        ]
                        <= 1.0
                    ),
                    "Convergence is within [-1, 1]",
                )

            self.check(
                f"{alert_id}: score validity",
                (
                    self.finite_number(score)
                    and score >= 0
                ),
                f"Score={score}",
            )

    def evaluate_incident_structure(self, incidents):
        incident_ids = [
            incident.get(
                "incident_id"
            )
            for incident in incidents
        ]

        self.check(
            "Unique incident IDs",
            len(incident_ids)
            == len(set(incident_ids)),
            (
                f"{len(incident_ids)} incidents inspected"
            ),
        )

        for incident in incidents:
            incident_id = incident.get(
                "incident_id",
                "UNKNOWN",
            )

            lifecycle = incident.get(
                "lifecycle",
                {},
            )

            state = lifecycle.get(
                "state"
            )

            self.check(
                f"{incident_id}: lifecycle state",
                state in {
                    "CONFIRMED",
                    "ACTIVE",
                    "RESOLVED",
                },
                f"State={state}",
            )

            classification = incident.get(
                "classification",
                {},
            )

            risk = classification.get(
                "risk"
            )

            alert_level = classification.get(
                "alert_level"
            )

            source_alert = incident.get(
                "source_alert",
                {},
            )

            source_alert_id = source_alert.get(
                "alert_id"
            )

            source_alert_level = source_alert.get(
                "alert_level"
            )

            source_alert_risk = source_alert.get(
                "risk",
                {},
            ).get(
                "peak_risk"
            )

            expected_source_alert_level = (
                EXPECTED_ALERT_LEVELS.get(
                    source_alert_risk
                )
            )

            self.check(
                f"{incident_id}: incident alert mapping",
                (
                    bool(source_alert_id)
                    and source_alert_level
                    == expected_source_alert_level
                ),
                (
                    f"Source alert risk="
                    f"{source_alert_risk}, "
                    f"expected="
                    f"{expected_source_alert_level}, "
                    f"actual="
                    f"{source_alert_level}; "
                    f"final incident risk="
                    f"{risk}, alert level="
                    f"{alert_level}; "
                    f"final classification may evolve "
                    f"after alert confirmation"
                ),
            )

            event_timing = incident.get(
                "event_timing",
                {},
            )

            duration = event_timing.get(
                "duration_seconds"
            )

            self.check(
                f"{incident_id}: duration",
                (
                    self.finite_number(duration)
                    and duration >= 0
                ),
                f"Duration={duration}",
            )

            confirmation = incident.get(
                "confirmation_evidence",
                {},
            )

            strong_frames = confirmation.get(
                "strong_frames",
                [],
            )

            strong_observations = confirmation.get(
                "strong_observations",
                0,
            )

            self.check(
                f"{incident_id}: strong observation count",
                (
                    strong_observations >= 3
                    and len(strong_frames)
                    >= strong_observations
                ),
                (
                    f"Strong observations="
                    f"{strong_observations}, "
                    f"strong frames="
                    f"{len(strong_frames)}"
                ),
            )

            source_alert = incident.get(
                "source_alert",
                {},
            )

            self.check(
                f"{incident_id}: source alert",
                bool(
                    source_alert.get(
                        "alert_id"
                    )
                ),
                (
                    f"Source alert="
                    f"{source_alert.get('alert_id')}"
                ),
            )

    def evaluate_alert_incident_mapping(
        self,
        alerts,
        incidents,
    ):
        alerts_by_id = {
            alert["alert_id"]: alert
            for alert in alerts
            if "alert_id" in alert
        }

        incidents_by_alert = {}

        for incident in incidents:
            alert_id = incident.get(
                "source_alert",
                {},
            ).get(
                "alert_id"
            )

            if alert_id:
                incidents_by_alert[
                    alert_id
                ] = incident

        for alert_id, alert in alerts_by_id.items():

            incident = incidents_by_alert.get(
                alert_id
            )

            self.check(
                f"{alert_id}: incident mapping",
                incident is not None,
                (
                    "Every live alert should "
                    "produce a corresponding incident"
                ),
            )

            if incident is None:
                continue

            alert_pair = self.pair_from_alert(
                alert
            )

            incident_pair = (
                self.pair_from_incident(
                    incident
                )
            )

            self.check(
                f"{alert_id}: pair consistency",
                alert_pair == incident_pair,
                (
                    f"Alert pair={alert_pair}, "
                    f"incident pair={incident_pair}"
                ),
            )

            source_alert_id = incident.get(
                "source_alert",
                {},
            ).get(
                "alert_id"
            )

            self.check(
                f"{alert_id}: source alert identity",
                source_alert_id == alert_id,
                (
                    f"Incident source alert="
                    f"{source_alert_id}"
                ),
            )

            alert_risk = alert.get(
                "risk",
                {},
            ).get(
                "peak_risk"
            )

            incident_risk = incident.get(
                "classification",
                {},
            ).get(
                "risk"
            )

            self.check(
                f"{alert_id}: risk evidence preserved",
                (
                    alert_risk in {
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                    }
                    and incident_risk in {
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                    }
                ),
                (
                    f"Alert risk={alert_risk}, "
                    f"final incident risk={incident_risk}; "
                    f"final classification may evolve "
                    f"after alert confirmation"
                ),
            )

            alert_score = alert.get(
                "risk",
                {},
            ).get(
                "peak_score"
            )

            incident_score = incident.get(
                "classification",
                {},
            ).get(
                "score"
            )

            self.check(
                f"{alert_id}: score evidence preserved",
                (
                    self.finite_number(
                        alert_score
                    )
                    and self.finite_number(
                        incident_score
                    )
                    and alert_score >= 0
                    and incident_score >= 0
                ),
                (
                    f"Alert score={alert_score}, "
                    f"final incident score={incident_score}; "
                    f"final score may increase as the "
                    f"incident continues"
                ),
            )

    def evaluate_analytics(
        self,
        incidents,
        analytics,
    ):
        overview = analytics.get(
            "overview",
            {},
        )

        total = overview.get(
            "total_incidents"
        )

        self.check(
            "Analytics incident total",
            total == len(incidents),
            (
                f"Analytics={total}, "
                f"incidents={len(incidents)}"
            ),
        )

        calculated_risk_counts = {}

        for incident in incidents:
            risk = incident.get(
                "classification",
                {},
            ).get(
                "risk"
            )

            calculated_risk_counts[
                risk
            ] = (
                calculated_risk_counts.get(
                    risk,
                    0,
                )
                + 1
            )

        analytics_risk_counts = overview.get(
            "risk_counts",
            {},
        )

        normalized_calculated_risk = {
            key: value
            for key, value
            in calculated_risk_counts.items()
            if value != 0
        }

        normalized_analytics_risk = {
            key: value
            for key, value
            in analytics_risk_counts.items()
            if value != 0
        }

        self.check(
            "Analytics risk counts",
            normalized_calculated_risk
            == normalized_analytics_risk,
            (
                f"Calculated="
                f"{normalized_calculated_risk}, "
                f"analytics="
                f"{normalized_analytics_risk}"
            ),
        )

        calculated_alert_counts = {}

        for incident in incidents:
            level = incident.get(
                "classification",
                {},
            ).get(
                "alert_level"
            )

            calculated_alert_counts[
                level
            ] = (
                calculated_alert_counts.get(
                    level,
                    0,
                )
                + 1
            )

        analytics_alert_counts = (
            overview.get(
                "alert_level_counts",
                {},
            )
        )

        normalized_calculated_alerts = {
            key: value
            for key, value
            in calculated_alert_counts.items()
            if value != 0
        }

        normalized_analytics_alerts = {
            key: value
            for key, value
            in analytics_alert_counts.items()
            if value != 0
        }

        self.check(
            "Analytics alert counts",
            normalized_calculated_alerts
            == normalized_analytics_alerts,
            (
                f"Calculated="
                f"{normalized_calculated_alerts}, "
                f"analytics="
                f"{normalized_analytics_alerts}"
            ),
        )

        calculated_states = {}

        for incident in incidents:
            state = incident.get(
                "lifecycle",
                {},
            ).get(
                "state"
            )

            calculated_states[
                state
            ] = (
                calculated_states.get(
                    state,
                    0,
                )
                + 1
            )

        analytics_states = overview.get(
            "lifecycle_state_counts",
            {},
        )

        normalized_calculated_states = {
            key: value
            for key, value
            in calculated_states.items()
            if value != 0
        }

        normalized_analytics_states = {
            key: value
            for key, value
            in analytics_states.items()
            if value != 0
        }

        self.check(
            "Analytics lifecycle counts",
            normalized_calculated_states
            == normalized_analytics_states,
            (
                f"Calculated="
                f"{normalized_calculated_states}, "
                f"analytics="
                f"{normalized_analytics_states}"
            ),
        )

        calculated_objects = {}

        for incident in incidents:
            pair = incident.get(
                "objects",
                {},
            ).get(
                "pair",
                [],
            )

            for object_id in pair:
                key = str(object_id)

                calculated_objects[
                    key
                ] = (
                    calculated_objects.get(
                        key,
                        0,
                    )
                    + 1
                )

        analytics_objects = analytics.get(
            "object_involvement",
            {},
        )

        self.check(
            "Analytics object involvement",
            calculated_objects
            == analytics_objects,
            (
                f"Calculated="
                f"{calculated_objects}, "
                f"analytics="
                f"{analytics_objects}"
            ),
        )

    def evaluate(self):
        print("=" * 70)
        print("INTELLIGENT ROAD SAFETY AI")
        print("SYSTEM EVALUATION & VALIDATION")
        print("=" * 70)

        alerts = self.load_json(
            ALERTS_PATH
        )

        incidents = self.load_json(
            INCIDENTS_PATH
        )

        analytics = self.load_json(
            ANALYTICS_PATH
        )

        alert_records = alerts.get(
            "alerts",
            []
        )

        incident_records = incidents.get(
            "incidents",
            []
        )

        self.evaluate_alert_structure(
            alert_records
        )

        self.evaluate_incident_structure(
            incident_records
        )

        self.evaluate_alert_incident_mapping(
            alert_records,
            incident_records,
        )

        self.evaluate_analytics(
            incident_records,
            analytics,
        )

        passed = sum(
            1
            for check in self.checks
            if check["status"] == "PASS"
        )

        failed = sum(
            1
            for check in self.checks
            if check["status"] == "FAIL"
        )

        warnings = sum(
            1
            for check in self.checks
            if check["status"] == "WARN"
        )

        total = len(
            self.checks
        )

        if failed > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "PASS_WITH_WARNINGS"
        else:
            overall_status = "PASS"

        output = {
            "version": "Safety-Evaluation-1.0",
            "evaluation_type": (
                "Internal system consistency validation"
            ),
            "sources": {
                "alerts": str(
                    ALERTS_PATH
                ),
                "incidents": str(
                    INCIDENTS_PATH
                ),
                "analytics": str(
                    ANALYTICS_PATH
                ),
            },
            "summary": {
                "overall_status": overall_status,
                "total_checks": total,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "pass_percentage": round(
                    (
                        passed / total * 100
                    )
                    if total
                    else 0.0,
                    2,
                ),
            },
            "checks": self.checks,
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=2,
            )

        print(
            f"Total checks: {total}"
        )

        print(
            f"Passed: {passed}"
        )

        print(
            f"Warnings: {warnings}"
        )

        print(
            f"Failed: {failed}"
        )

        print(
            f"Overall status: "
            f"{overall_status}"
        )

        print(
            f"\nOutput: {OUTPUT_PATH}"
        )

        print("=" * 70)

        return output


def main():
    evaluator = SafetyEvaluation()

    try:
        evaluator.evaluate()
    except Exception as error:
        print(
            "\nEvaluation failed:"
        )
        print(error)
        raise


if __name__ == "__main__":
    main()