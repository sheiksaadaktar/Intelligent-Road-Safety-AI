import json
from pathlib import Path
from html import escape


# ============================================================
# INTELLIGENT ROAD SAFETY AI
# ALERT MONITOR
# ============================================================

INPUT_PATH = Path("output/alerts_v57.json")
OUTPUT_PATH = Path("output/alert_monitor.html")


def load_alerts():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"V5.7 alert file not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as alert_file:
        return json.load(alert_file)


def fmt(value, decimals=2):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.{decimals}f}"

    return str(value)


def alert_class(alert_level):
    mapping = {
        "IMMEDIATE_ALERT": "immediate",
        "PRIORITY_ALERT": "priority",
        "MONITOR": "monitor",
    }

    return mapping.get(
        alert_level,
        "monitor",
    )


def build_summary_cards(statistics):
    counts = statistics.get(
        "alert_counts",
        {},
    )

    return f"""
    <section class="summary-grid">

        <div class="summary-card total">
            <div class="summary-label">Total Alerts</div>
            <div class="summary-value">
                {statistics.get("total_alerts", 0)}
            </div>
        </div>

        <div class="summary-card immediate">
            <div class="summary-label">Immediate Alerts</div>
            <div class="summary-value">
                {counts.get("IMMEDIATE_ALERT", 0)}
            </div>
        </div>

        <div class="summary-card priority">
            <div class="summary-label">Priority Alerts</div>
            <div class="summary-value">
                {counts.get("PRIORITY_ALERT", 0)}
            </div>
        </div>

        <div class="summary-card monitor">
            <div class="summary-label">Monitoring</div>
            <div class="summary-value">
                {counts.get("MONITOR", 0)}
            </div>
        </div>

    </section>
    """


def build_alert_cards(alerts):
    if not alerts:
        return """
        <div class="empty-state">
            No safety alerts are currently available.
        </div>
        """

    cards = []

    for alert in alerts:
        level = escape(
            str(
                alert.get(
                    "alert_level",
                    "MONITOR",
                )
            )
        )

        css_class = alert_class(level)

        pair = alert.get(
            "pair",
            [],
        )

        pair_text = " ↔ ".join(
            str(item)
            for item in pair
        )

        risk = alert.get(
            "risk",
            {},
        )

        event = alert.get(
            "event",
            {},
        )

        trajectory = alert.get(
            "trajectory_evidence",
            {},
        )

        confirmation = alert.get(
            "confirmation",
            {},
        )

        cards.append(
            f"""
            <article class="alert-card {css_class}">

                <div class="alert-header">

                    <div>
                        <div class="alert-id">
                            {escape(
                                str(
                                    alert.get(
                                        "alert_id",
                                        "N/A",
                                    )
                                )
                            )}
                        </div>

                        <div class="object-pair">
                            Object {escape(pair_text)}
                        </div>
                    </div>

                    <div class="alert-badge">
                        {level}
                    </div>

                </div>

                <div class="alert-message">
                    {escape(
                        str(
                            alert.get(
                                "message",
                                "",
                            )
                        )
                    )}
                </div>

                <div class="metric-grid">

                    <div class="metric">
                        <span>Risk</span>
                        <strong>
                            {escape(
                                str(
                                    risk.get(
                                        "peak_risk",
                                        "N/A",
                                    )
                                )
                            )}
                        </strong>
                    </div>

                    <div class="metric">
                        <span>Score</span>
                        <strong>
                            {fmt(
                                risk.get(
                                    "peak_score"
                                ),
                                0,
                            )}
                        </strong>
                    </div>

                    <div class="metric">
                        <span>Priority</span>
                        <strong>
                            {fmt(
                                alert.get(
                                    "priority"
                                ),
                                0,
                            )}
                        </strong>
                    </div>

                    <div class="metric">
                        <span>State</span>
                        <strong>
                            {escape(
                                str(
                                    alert.get(
                                        "state",
                                        "N/A",
                                    )
                                )
                            )}
                        </strong>
                    </div>

                </div>

                <div class="section-title">
                    Event
                </div>

                <div class="detail-grid">

                    <div>
                        <span>Start frame</span>
                        <strong>
                            {fmt(
                                event.get(
                                    "start_frame"
                                ),
                                0,
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>Last frame</span>
                        <strong>
                            {fmt(
                                event.get(
                                    "last_frame"
                                ),
                                0,
                            )}
                        </strong>
                    </div>

                    <div>
                        <span>Duration</span>
                        <strong>
                            {fmt(
                                event.get(
                                    "duration_seconds"
                                )
                            )} s
                        </strong>
                    </div>

                </div>

                <div class="section-title">
                    Trajectory Evidence
                </div>

                <div class="detail-grid">

                    <div>
                        <span>Min distance</span>
                        <strong>
                            {fmt(
                                trajectory.get(
                                    "min_distance_pixels"
                                )
                            )} px
                        </strong>
                    </div>

                    <div>
                        <span>Predicted distance</span>
                        <strong>
                            {fmt(
                                trajectory.get(
                                    "min_predicted_distance_pixels"
                                )
                            )} px
                        </strong>
                    </div>

                    <div>
                        <span>Min TCA</span>
                        <strong>
                            {fmt(
                                trajectory.get(
                                    "min_tca_seconds"
                                )
                            )} s
                        </strong>
                    </div>

                    <div>
                        <span>Approach speed</span>
                        <strong>
                            {fmt(
                                trajectory.get(
                                    "max_approach_speed_pixels_per_second"
                                )
                            )} px/s
                        </strong>
                    </div>

                    <div>
                        <span>Convergence</span>
                        <strong>
                            {fmt(
                                trajectory.get(
                                    "max_convergence"
                                )
                            )}
                        </strong>
                    </div>

                </div>

                <div class="section-title">
                    Confirmation
                </div>

                <div class="confirmation-row">

                    <span>
                        Frames observed:
                        <strong>
                            {fmt(
                                confirmation.get(
                                    "frames_seen"
                                ),
                                0,
                            )}
                        </strong>
                    </span>

                    <span>
                        Strong observations:
                        <strong>
                            {fmt(
                                confirmation.get(
                                    "strong_observations"
                                ),
                                0,
                            )}
                        </strong>
                    </span>

                    <span>
                        Confirmed:
                        <strong>
                            {str(
                                confirmation.get(
                                    "confirmed",
                                    False,
                                )
                            ).upper()}
                        </strong>
                    </span>

                </div>

            </article>
            """
        )

    return "\n".join(cards)


def build_html(data):
    statistics = data.get(
        "statistics",
        {},
    )

    alerts = data.get(
        "alerts",
        [],
    )

    summary_cards = build_summary_cards(
        statistics
    )

    alert_cards = build_alert_cards(
        alerts
    )

    source_video = escape(
        str(
            data.get(
                "source_video",
                "N/A",
            )
        )
    )

    source_version = escape(
        str(
            data.get(
                "source_version",
                "N/A",
            )
        )
    )

    fps = fmt(
        data.get(
            "fps"
        ),
        2,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Intelligent Road Safety AI - Alert Monitor
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Inter,
        Segoe UI,
        Arial,
        sans-serif;
    background: #0b1220;
    color: #e5e7eb;
}}

.container {{
    width: min(1500px, 94%);
    margin: 0 auto;
}}

.header {{
    padding: 32px 0 24px;
    border-bottom: 1px solid #263244;
}}

.header h1 {{
    margin: 0;
    font-size: 30px;
    letter-spacing: 0.4px;
}}

.header p {{
    margin: 8px 0 0;
    color: #94a3b8;
}}

.summary-grid {{
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin: 24px 0;
}}

.summary-card {{
    background: #111a2b;
    border: 1px solid #263244;
    border-radius: 14px;
    padding: 20px;
}}

.summary-card.immediate {{
    border-left: 5px solid #ef4444;
}}

.summary-card.priority {{
    border-left: 5px solid #f59e0b;
}}

.summary-card.monitor {{
    border-left: 5px solid #3b82f6;
}}

.summary-card.total {{
    border-left: 5px solid #94a3b8;
}}

.summary-label {{
    color: #94a3b8;
    font-size: 14px;
}}

.summary-value {{
    font-size: 34px;
    font-weight: 700;
    margin-top: 8px;
}}

.section-heading {{
    margin: 30px 0 16px;
}}

.section-heading h2 {{
    margin: 0;
    font-size: 22px;
}}

.section-heading p {{
    margin: 6px 0 0;
    color: #94a3b8;
}}

.alert-list {{
    display: grid;
    gap: 18px;
}}

.alert-card {{
    background: #111a2b;
    border: 1px solid #263244;
    border-radius: 14px;
    padding: 22px;
}}

.alert-card.immediate {{
    border-left: 6px solid #ef4444;
}}

.alert-card.priority {{
    border-left: 6px solid #f59e0b;
}}

.alert-card.monitor {{
    border-left: 6px solid #3b82f6;
}}

.alert-header {{
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: flex-start;
}}

.alert-id {{
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

.object-pair {{
    font-size: 22px;
    font-weight: 700;
    margin-top: 5px;
}}

.alert-badge {{
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}}

.immediate .alert-badge {{
    background: #451a1a;
    color: #fca5a5;
}}

.priority .alert-badge {{
    background: #422006;
    color: #fcd34d;
}}

.monitor .alert-badge {{
    background: #172554;
    color: #93c5fd;
}}

.alert-message {{
    margin-top: 16px;
    color: #cbd5e1;
}}

.metric-grid {{
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 18px;
}}

.metric {{
    background: #0b1220;
    border-radius: 10px;
    padding: 12px;
}}

.metric span,
.detail-grid span {{
    display: block;
    color: #64748b;
    font-size: 12px;
    margin-bottom: 5px;
}}

.metric strong,
.detail-grid strong {{
    font-size: 15px;
}}

.section-title {{
    margin-top: 20px;
    margin-bottom: 10px;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

.detail-grid {{
    display: grid;
    grid-template-columns:
        repeat(5, minmax(0, 1fr));
    gap: 10px;
}}

.detail-grid > div {{
    background: #0b1220;
    border-radius: 10px;
    padding: 12px;
}}

.confirmation-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    color: #94a3b8;
    font-size: 13px;
}}

.confirmation-row strong {{
    color: #e5e7eb;
    margin-left: 5px;
}}

.empty-state {{
    padding: 40px;
    text-align: center;
    background: #111a2b;
    border: 1px solid #263244;
    border-radius: 14px;
    color: #94a3b8;
}}

.metadata {{
    margin: 32px 0;
    padding: 20px;
    background: #111a2b;
    border: 1px solid #263244;
    border-radius: 14px;
}}

.metadata-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 16px;
}}

.metadata-item span {{
    display: block;
    color: #64748b;
    font-size: 12px;
    margin-bottom: 5px;
}}

.metadata-item strong {{
    font-size: 14px;
    word-break: break-word;
}}

footer {{
    padding: 24px 0 36px;
    color: #64748b;
    font-size: 12px;
    text-align: center;
}}

@media (max-width: 1000px) {{

    .summary-grid,
    .metric-grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .detail-grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .metadata-grid {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 600px) {{

    .summary-grid,
    .metric-grid,
    .detail-grid {{
        grid-template-columns: 1fr;
    }}

    .alert-header {{
        flex-direction: column;
    }}

}}

</style>

</head>

<body>

<div class="container">

<header class="header">

    <h1>
        Intelligent Road Safety AI
    </h1>

    <p>
        Live Safety Alert Monitor
    </p>

</header>

{summary_cards}

<section>

    <div class="section-heading">

        <h2>
            Safety Alerts
        </h2>

        <p>
            Alerts generated from the validated
            V5.7 safety alert pipeline.
        </p>

    </div>

    <div class="alert-list">

        {alert_cards}

    </div>

</section>

<section class="metadata">

    <div class="section-heading">

        <h2>
            System Information
        </h2>

    </div>

    <div class="metadata-grid">

        <div class="metadata-item">
            <span>Alert version</span>
            <strong>V5.7</strong>
        </div>

        <div class="metadata-item">
            <span>Source version</span>
            <strong>{source_version}</strong>
        </div>

        <div class="metadata-item">
            <span>Video FPS</span>
            <strong>{fps}</strong>
        </div>

        <div class="metadata-item">
            <span>Source video</span>
            <strong>{source_video}</strong>
        </div>

        <div class="metadata-item">
            <span>Total alerts</span>
            <strong>
                {statistics.get("total_alerts", 0)}
            </strong>
        </div>

        <div class="metadata-item">
            <span>Active states</span>
            <strong>
                {statistics.get(
                    "state_counts",
                    {}
                ).get("ACTIVE", 0)}
            </strong>
        </div>

    </div>

</section>

<footer>
    Intelligent Road Safety AI —
    Safety Alert Monitoring Interface
</footer>

</div>

</body>
</html>
"""


def main():
    print("=" * 70)
    print("INTELLIGENT ROAD SAFETY AI")
    print("ALERT MONITOR")
    print("=" * 70)

    print(
        f"Input: {INPUT_PATH}"
    )

    data = load_alerts()

    alerts = data.get(
        "alerts",
        [],
    )

    print(
        f"Alerts loaded: {len(alerts)}"
    )

    html = build_html(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(html)

    print("")
    print(
        f"Alert monitor exported to: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
