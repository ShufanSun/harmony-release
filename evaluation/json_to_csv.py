"""Flatten eval_runner.py's JSON output into a wide CSV (one row per scene).

Columns:
    scene, <method>_n_clip, <method>_pl, ..., best_n_clip_method, best_pl_method

Empty cells = method has no result for that scene.

Usage:
    python json_to_csv.py [results.json] [out.csv]
"""
import csv
import json
import sys


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/results_all.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".json", ".csv")

    with open(in_path) as f:
        data = json.load(f)

    methods = list(data["methods"].keys())

    # Collect all scenes across methods
    all_scenes: set[str] = set()
    for m in methods:
        per = data["methods"][m].get("per_scene", {})
        all_scenes.update(per.keys())
    scenes = sorted(all_scenes)

    # Build rows
    header = ["scene"]
    for m in methods:
        header += [f"{m}_n_clip", f"{m}_pl"]
    header += ["best_n_clip_method", "best_pl_method"]

    rows = []
    for s in scenes:
        row: dict = {"scene": s}
        n_clips: dict[str, float] = {}
        pls: dict[str, float] = {}
        for m in methods:
            entry = data["methods"][m].get("per_scene", {}).get(s, {})
            if "n_clip" in entry and "pl" in entry:
                row[f"{m}_n_clip"] = f"{entry['n_clip']:.4f}"
                row[f"{m}_pl"]     = f"{entry['pl']:.4f}"
                n_clips[m] = entry["n_clip"]
                pls[m]     = entry["pl"]
            else:
                row[f"{m}_n_clip"] = ""
                row[f"{m}_pl"]     = ""
        row["best_n_clip_method"] = min(n_clips, key=n_clips.get) if n_clips else ""
        row["best_pl_method"]     = min(pls,     key=pls.get)     if pls     else ""
        rows.append(row)

    # Build averages row
    avg_row = {"scene": "AVERAGE"}
    for m in methods:
        avgs = data["methods"][m].get("averages", {})
        avg_row[f"{m}_n_clip"] = f"{avgs['avg_n_clip']:.4f}" if avgs else ""
        avg_row[f"{m}_pl"]     = f"{avgs['avg_pl']:.4f}"     if avgs else ""
    avg_row["best_n_clip_method"] = ""
    avg_row["best_pl_method"]     = ""

    # Build counts row
    count_row = {"scene": "NUM_SCORED"}
    for m in methods:
        avgs = data["methods"][m].get("averages", {})
        count_row[f"{m}_n_clip"] = avgs.get("num_scored", "") if avgs else ""
        count_row[f"{m}_pl"]     = avgs.get("num_scored", "") if avgs else ""
    count_row["best_n_clip_method"] = ""
    count_row["best_pl_method"]     = ""

    # Intersection averages — only over scenes where ALL methods have a result
    def valid_scenes(m):
        per = data["methods"][m].get("per_scene", {})
        return {k for k, v in per.items() if "n_clip" in v and "pl" in v}
    intersection_set = set.intersection(*(valid_scenes(m) for m in methods)) if methods else set()

    intersect_row = {"scene": f"AVERAGE_INTERSECT({len(intersection_set)})"}
    intersect_n_row = {"scene": "INTERSECT_NUM"}
    for m in methods:
        per = data["methods"][m].get("per_scene", {})
        n_clips = [per[s]["n_clip"] for s in intersection_set]
        pls     = [per[s]["pl"]     for s in intersection_set]
        if n_clips:
            intersect_row[f"{m}_n_clip"] = f"{sum(n_clips)/len(n_clips):.4f}"
            intersect_row[f"{m}_pl"]     = f"{sum(pls)/len(pls):.4f}"
        else:
            intersect_row[f"{m}_n_clip"] = ""
            intersect_row[f"{m}_pl"]     = ""
        intersect_n_row[f"{m}_n_clip"] = len(intersection_set)
        intersect_n_row[f"{m}_pl"]     = len(intersection_set)
    intersect_row["best_n_clip_method"] = ""
    intersect_row["best_pl_method"]     = ""
    intersect_n_row["best_n_clip_method"] = ""
    intersect_n_row["best_pl_method"]     = ""

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({})  # blank line
        w.writerow(avg_row)
        w.writerow(count_row)
        w.writerow({})
        w.writerow(intersect_row)
        w.writerow(intersect_n_row)

    print(f"wrote {len(rows)} scene rows + averages → {out_path}")
    print(f"methods: {methods}")


if __name__ == "__main__":
    main()
