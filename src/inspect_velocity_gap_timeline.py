import csv
from pathlib import Path

path = Path("output/velocity_risk_comparison/pairwise_risk_comparison.csv")

targets = {
    ("pytorch", 4, 11),
    ("openvino", 33, 40),
}

columns = [
    "backend", "frame", "id_1", "id_2",
    "original_status", "experimental_status",
    "original_risk", "experimental_risk",
    "original_score", "experimental_score",
    "original_approach_speed", "experimental_approach_speed",
    "original_predicted_distance", "experimental_predicted_distance",
    "original_tca", "experimental_tca",
    "original_convergence", "experimental_convergence",
]

with path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("========== TARGET PAIR TIMELINE: FRAMES 270-345 ==========")

selected = [
    r for r in rows
    if (
        r["backend"],
        int(r["id_1"]),
        int(r["id_2"]),
    ) in targets
    and 270 <= int(r["frame"]) <= 345
]

for row in sorted(selected, key=lambda r: (r["backend"], int(r["frame"]))):
    print("\n" + " | ".join(f"{c}={row.get(c, '')}" for c in columns))

print("\n========== ALL NO-HISTORY STATUS DIFFERENCES ==========")

no_history = [
    r for r in rows
    if r["original_status"] == "EVALUATED"
    and r["experimental_status"] == "NO_VELOCITY_HISTORY"
]

print("Total:", len(no_history))

for backend in ("pytorch", "openvino"):
    backend_rows = [r for r in no_history if r["backend"] == backend]
    grouped = {}

    for row in backend_rows:
        key = (row["id_1"], row["id_2"])
        grouped.setdefault(key, []).append(int(row["frame"]))

    print(f"\n{backend.upper()}: {len(backend_rows)} affected pair-frames")
    for (id_1, id_2), frames in sorted(grouped.items()):
        print(
            f"IDs {id_1}-{id_2}: {len(frames)} frames, "
            f"range {min(frames)}-{max(frames)}"
        )

print("\nDiagnostic complete. No production files were modified.")
