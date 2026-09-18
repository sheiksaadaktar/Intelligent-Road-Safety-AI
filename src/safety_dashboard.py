import json
from pathlib import Path
from html import escape


INPUT_PATH = Path("output/safety_report_v54.json")
OUTPUT_PATH = Path("output/safety_dashboard_v55.html")


def load_report():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"V5.4 report not found: {INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as report_file:
        return json.load(report_file)


def fmt(value, digits=2):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def severity_class(severity):
    return {
        "CRITICAL": "critical",
        "HIGH": "high",
        "MEDIUM": "medium",
        "LOW": "low",
    }.get(severity, "unknown")


def build_severity_cards(stats):
    counts = stats["severity_counts"]
    percentages = stats["severity_percentages"]

    cards = []

    for severity in ("CRITICAL", "HIGH", "MEDIUM"):
        cards.append(
            f"""
            <div class="severity-card {severity.lower()}">
                <div class="severity-name">{severity}</div>
                <div class="severity-count">
                    {counts.get(severity, 0)}
                </div>
                <div class="severity-percent">
                    {fmt(percentages.get(severity, 0), 1)}%
                </div>
            </div>
            """
        )

    return "\n".join(cards)


def build_object_rows(stats):
    involvement = stats.get("object_involvement", {})

    sorted_objects = sorted(
        involvement.items(),
        key=lambda item: (-item[1], int(item[0])),
    )

    rows = []

    for object_id, count in sorted_objects:
        rows.append(
            f"""
            <tr>
                <td>ID {escape(str(object_id))}</td>
                <td>{count}</td>
            </tr>
            """
        )

    if not rows:
        return """
        <tr>
            <td colspan="2">No object involvement data.</td>
        </tr>
        """

    return "\n".join(rows)


def build_event_rows(events):
    rows = []

    for index, event in enumerate(events, start=1):
        pair = event.get("pair", [])
        pair_text = " ↔ ".join(
            f"ID {escape(str(object_id))}"
            for object_id in pair
        )

        severity = event.get(
            "peak_risk",
            "UNKNOWN",
        )

        rows.append(
            f"""
            <tr>
                <td>{index}</td>

                <td>
                    <strong>{pair_text}</strong>
                </td>

                <td>
                    <span class="badge {severity_class(severity)}">
                        {escape(str(severity))}
                    </span>
                </td>

                <td>{event.get("peak_score", "N/A")}</td>

                <td>
                    {event.get("start_frame", "N/A")}
                    –
                    {event.get("last_frame", "N/A")}
                </td>

                <td>
                    {event.get("frames_seen", "N/A")}
                </td>

                <td>
                    {fmt(event.get("duration_seconds"))}
                </td>

                <td>
                    {fmt(event.get("min_distance"))}
                </td>

                <td>
                    {fmt(event.get("min_predicted_distance"))}
                </td>

                <td>
                    {fmt(event.get("min_tca"))}
                </td>

                <td>
                    {fmt(event.get("max_approach_speed"))}
                </td>

                <td>
                    {fmt(event.get("max_convergence"))}
                </td>
            </tr>
            """
        )

    if not rows:
        return """
        <tr>
            <td colspan="12">No confirmed risk events.</td>
        </tr>
        """

    return "\n".join(rows)


def build_html(report):
    stats = report["statistics"]

    total_events = stats["total_confirmed_events"]

    duration = stats["duration"]
    distance = stats["distance"]
    predicted = stats["predicted_distance"]
    tca = stats["tca"]
    approach = stats["approach_speed"]
    convergence = stats["convergence"]

    severity_cards = build_severity_cards(stats)
    object_rows = build_object_rows(stats)
    event_rows = build_event_rows(
        report.get("events", [])
    )

    video_path = escape(
        str(report.get("video_path", "N/A"))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    Intelligent Road Safety AI - V5.5 Safety Dashboard
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
    background: #f4f6f8;
    color: #1f2933;
}}

header {{
    background: #111827;
    color: white;
    padding: 28px 34px;
}}

header h1 {{
    margin: 0 0 8px 0;
    font-size: 28px;
}}

header p {{
    margin: 0;
    color: #cbd5e1;
}}

.container {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 28px;
}}

.section {{
    margin-bottom: 28px;
}}

.section-title {{
    font-size: 20px;
    margin-bottom: 14px;
}}

.overview {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
}}

.metric-card {{
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow:
        0 2px 8px rgba(0,0,0,0.08);
}}

.metric-label {{
    font-size: 13px;
    color: #64748b;
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 27px;
    font-weight: 700;
}}

.metric-unit {{
    font-size: 13px;
    color: #64748b;
}}

.severity-grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
}}

.severity-card {{
    background: white;
    border-radius: 10px;
    padding: 20px;
    border-left: 6px solid #94a3b8;
    box-shadow:
        0 2px 8px rgba(0,0,0,0.08);
}}

.severity-card.critical {{
    border-left-color: #b91c1c;
}}

.severity-card.high {{
    border-left-color: #ea580c;
}}

.severity-card.medium {{
    border-left-color: #ca8a04;
}}

.severity-name {{
    font-weight: 700;
    font-size: 14px;
}}

.severity-count {{
    font-size: 32px;
    font-weight: 700;
    margin-top: 6px;
}}

.severity-percent {{
    color: #64748b;
    font-size: 13px;
}}

.two-column {{
    display: grid;
    grid-template-columns:
        minmax(0, 2fr)
        minmax(260px, 1fr);
    gap: 20px;
}}

.panel {{
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow:
        0 2px 8px rgba(0,0,0,0.08);
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th {{
    background: #e5e7eb;
    text-align: left;
    padding: 10px;
    font-size: 12px;
    white-space: nowrap;
}}

td {{
    border-bottom: 1px solid #e5e7eb;
    padding: 10px;
    font-size: 12px;
    white-space: nowrap;
}}

tbody tr:hover {{
    background: #f8fafc;
}}

.table-wrapper {{
    overflow-x: auto;
}}

.badge {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
}}

.badge.critical {{
    background: #fee2e2;
    color: #991b1b;
}}

.badge.high {{
    background: #ffedd5;
    color: #9a3412;
}}

.badge.medium {{
    background: #fef3c7;
    color: #92400e;
}}

.badge.low {{
    background: #dcfce7;
    color: #166534;
}}

.badge.unknown {{
    background: #e5e7eb;
    color: #374151;
}}

.metadata {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}}

.metadata-item {{
    background: #f8fafc;
    padding: 12px;
    border-radius: 7px;
}}

.metadata-label {{
    color: #64748b;
    font-size: 12px;
}}

.metadata-value {{
    margin-top: 4px;
    font-weight: 600;
    word-break: break-word;
}}

footer {{
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 24px;
}}

@media (max-width: 800px) {{
    .two-column {{
        grid-template-columns: 1fr;
    }}

    header {{
        padding: 22px;
    }}

    .container {{
        padding: 18px;
    }}
}}

</style>
</head>

<body>

<header>
    <h1>Intelligent Road Safety AI</h1>
    <p>V5.5 Safety Intelligence Dashboard</p>
</header>

<div class="container">

<section class="section">

    <div class="section-title">
        Overview
    </div>

    <div class="overview">

        <div class="metric-card">
            <div class="metric-label">
                Confirmed Risk Events
            </div>
            <div class="metric-value">
                {total_events}
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">
                Average Event Duration
            </div>
            <div class="metric-value">
                {fmt(duration["average_seconds"])}
            </div>
            <div class="metric-unit">
                seconds
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">
                Minimum Distance
            </div>
            <div class="metric-value">
                {fmt(distance["minimum_pixels"])}
            </div>
            <div class="metric-unit">
                pixels
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">
                Minimum Predicted Distance
            </div>
            <div class="metric-value">
                {fmt(predicted["minimum_pixels"])}
            </div>
            <div class="metric-unit">
                pixels
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">
                Minimum TCA
            </div>
            <div class="metric-value">
                {fmt(tca["minimum_seconds"])}
            </div>
            <div class="metric-unit">
                seconds
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">
                Maximum Approach Speed
            </div>
            <div class="metric-value">
                {fmt(approach["maximum_pixels_per_second"])}
            </div>
            <div class="metric-unit">
                pixels / second
            </div>
        </div>

    </div>

</section>

<section class="section">

    <div class="section-title">
        Severity Distribution
    </div>

    <div class="severity-grid">
        {severity_cards}
    </div>

</section>

<section class="section">

    <div class="two-column">

        <div class="panel">

            <div class="section-title">
                Trajectory Statistics
            </div>

            <table>

                <tbody>

                    <tr>
                        <td>Average distance</td>
                        <td>
                            {fmt(distance["average_pixels"])}
                            px
                        </td>
                    </tr>

                    <tr>
                        <td>Average predicted distance</td>
                        <td>
                            {fmt(predicted["average_pixels"])}
                            px
                        </td>
                    </tr>

                    <tr>
                        <td>Average TCA</td>
                        <td>
                            {fmt(tca["average_seconds"])}
                            s
                        </td>
                    </tr>

                    <tr>
                        <td>Maximum approach speed</td>
                        <td>
                            {fmt(approach["maximum_pixels_per_second"])}
                            px/s
                        </td>
                    </tr>

                    <tr>
                        <td>Average approach speed</td>
                        <td>
                            {fmt(approach["average_pixels_per_second"])}
                            px/s
                        </td>
                    </tr>

                    <tr>
                        <td>Maximum convergence</td>
                        <td>
                            {fmt(convergence["maximum"])}
                        </td>
                    </tr>

                    <tr>
                        <td>Average convergence</td>
                        <td>
                            {fmt(convergence["average"])}
                        </td>
                    </tr>

                    <tr>
                        <td>Maximum event duration</td>
                        <td>
                            {fmt(duration["maximum_seconds"])}
                            s
                        </td>
                    </tr>

                    <tr>
                        <td>Minimum event duration</td>
                        <td>
                            {fmt(duration["minimum_seconds"])}
                            s
                        </td>
                    </tr>

                </tbody>

            </table>

        </div>

        <div class="panel">

            <div class="section-title">
                Object Involvement
            </div>

            <table>

                <thead>
                    <tr>
                        <th>Object</th>
                        <th>Events</th>
                    </tr>
                </thead>

                <tbody>
                    {object_rows}
                </tbody>

            </table>

        </div>

    </div>

</section>

<section class="section">

    <div class="panel">

        <div class="section-title">
            Confirmed Risk Events
        </div>

        <div class="table-wrapper">

            <table>

                <thead>

                    <tr>
                        <th>#</th>
                        <th>Pair</th>
                        <th>Peak Risk</th>
                        <th>Score</th>
                        <th>Frames</th>
                        <th>Observed</th>
                        <th>Duration (s)</th>
                        <th>Min Distance</th>
                        <th>Min Predicted</th>
                        <th>Min TCA</th>
                        <th>Max Approach</th>
                        <th>Max Convergence</th>
                    </tr>

                </thead>

                <tbody>
                    {event_rows}
                </tbody>

            </table>

        </div>

    </div>

</section>

<section class="section">

    <div class="panel">

        <div class="section-title">
            Report Metadata
        </div>

        <div class="metadata">

            <div class="metadata-item">
                <div class="metadata-label">
                    Report Version
                </div>
                <div class="metadata-value">
                    {escape(str(report.get("version", "N/A")))}
                </div>
            </div>

            <div class="metadata-item">
                <div class="metadata-label">
                    Source Version
                </div>
                <div class="metadata-value">
                    {escape(str(report.get("source_version", "N/A")))}
                </div>
            </div>

            <div class="metadata-item">
                <div class="metadata-label">
                    Video FPS
                </div>
                <div class="metadata-value">
                    {fmt(report.get("fps"))}
                </div>
            </div>

            <div class="metadata-item">
                <div class="metadata-label">
                    Source Video
                </div>
                <div class="metadata-value">
                    {video_path}
                </div>
            </div>

        </div>

    </div>

</section>

</div>

<footer>
    Intelligent Road Safety AI · V5.5
</footer>

</body>
</html>
"""


def main():
    print("=" * 70)
    print("V5.5 SAFETY DASHBOARD")
    print("=" * 70)

    report = load_report()

    print(
        f"Input: {INPUT_PATH}"
    )

    print(
        f"Confirmed events: "
        f"{report['statistics']['total_confirmed_events']}"
    )

    html = build_html(report)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as dashboard_file:
        dashboard_file.write(html)

    print(
        f"Dashboard exported to: {OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
