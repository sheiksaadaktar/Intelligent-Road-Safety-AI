import csv
from collections import Counter, defaultdict
from pathlib import Path

path = Path("output/velocity_risk_comparison/pairwise_risk_comparison.csv")

with path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

status_changes = []
risk_changes = []

for row in rows:
    original_status = row["original_status"]
    experimental_status = row["experimental_status"]

    if original_status != experimental_status:
        status_changes.append(row)

    if original_status == "EVALUATED" and experimental_status == "EVALUATED":
        if row["original_risk"] != row["experimental_risk"]:
            risk_changes.append(row)

print("\n========== STATUS DIFFERENCES ==========")
print("Total rows with different evaluation status:", len(status_changes))

status_counts = Counter(
    (r["original_status"], r["experimental_status"])
    for r in status_changes
)

for transition, count in sorted(status_counts.items()):
    print(f"{transition[0]} -> {transition[1]}: {count}")

print("\n========== RISK CLASSIFICATION CHANGES ==========")
print("Both methods evaluated, but risk changed:", len(risk_changes))

risk_counts = Counter(
    (r["original_risk"], r["experimental_risk"])
    for r in risk_changes
)

for transition, count in sorted(risk_counts.items()):
    print(f"{transition[0]} -> {transition[1]}: {count}")

print("\n========== INDIVIDUAL CHANGED OBSERVATIONS ==========")

columns = [
    "backend", "frame", "id_1", "id_2",
    "original_risk", "experimental_risk",
    "original_score", "experimental_score",
    "original_approach_speed", "experimental_approach_speed",
    "original_predicted_distance", "experimental_predicted_distance",
    "original_tca", "experimental_tca",
    "original_convergence", "experimental_convergence",
]

for row in sorted(
    risk_changes,
    key=lambda r: (r["backend"], int(r["frame"]), int(r["id_1"]), int(r["id_2"]))
):
    print("\n" + " | ".join(f"{c}={row.get(c, '')}" for c in columns))

print("\n========== CHANGES GROUPED BY OBJECT PAIR ==========")

grouped = defaultdict(list)

for row in risk_changes:
    key = (
        row["backend"],
        row["id_1"],
        row["id_2"],
        row["original_risk"],
        row["experimental_risk"],
    )
    grouped[key].append(int(row["frame"]))

for key, frames in sorted(grouped.items()):
    print(
        f"{key[0].upper()} IDs {key[1]}-{key[2]}: "
        f"{key[3]} -> {key[4]} | "
        f"{len(frames)} frames | "
        f"frames {min(frames)}-{max(frames)}"
    )

print("\nInspection complete. No project or production files were modified.")
