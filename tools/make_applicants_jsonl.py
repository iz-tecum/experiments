#!/usr/bin/env python3
import argparse
import ast
import csv
import json
from pathlib import Path


def first(row, *keys, default=""):
    """Return the first matching column value from row for any of the given keys."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def parse_features(s: str):
    s = (s or "").strip()
    if not s:
        return None
    # Most likely JSON like: [1,2,3] but sometimes Google Sheets exports odd quoting.
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass

    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Submissions CSV (export from Google Sheets)")
    ap.add_argument("--out", required=True, help="Output applicants.jsonl path")
    ap.add_argument("--id-mode", choices=["uni_timestamp", "row"], default="uni_timestamp")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f, out_path.open("w", encoding="utf-8") as out:
        r = csv.DictReader(f)
        for idx, row in enumerate(r, start=1):
            # Your sheet uses these names (from your screenshot):
            # timestamp, fullName, uni, email, schoolYear, raceEth, gpa, calc12, courses,
            # score_0_10, raw_score, feature_version, resume_chars, features_json, essayMath, essayCommunity, ...
            features_raw = first(row, "features_json", "features", "featuresJson", default="")
            feats = parse_features(features_raw)
            if not feats:
                n_skip += 1
                continue

            # Coerce to floats
            try:
                feats = [float(x) for x in feats]
            except Exception:
                n_skip += 1
                continue

            # Expect 21 features (your v2 schema)
            if len(feats) != 21:
                n_skip += 1
                continue

            timestamp = first(row, "timestamp", default="").strip()
            uni = first(row, "uni", default="").strip()
            full_name = first(row, "fullName", "fullname", "name", default="").strip()

            if args.id_mode == "uni_timestamp" and uni and timestamp:
                applicant_id = f"{uni}_{timestamp}"
            elif args.id_mode == "uni_timestamp" and uni:
                applicant_id = f"{uni}_{idx}"
            else:
                applicant_id = str(idx)

            meta = {
                "name": full_name,
                "uni": uni,
                "email": first(row, "email", default="").strip(),
                "schoolYear": first(row, "schoolYear", "schoolYearText", default="").strip(),
                "raceEth": first(row, "raceEth", default="").strip(),
                "gpa": first(row, "gpa", default="").strip(),
                "calc12": first(row, "calc12", "calcVal", default="").strip(),
                "courses": first(row, "courses", default="").strip(),
                "score_0_10": first(row, "score_0_10", default="").strip(),
                "raw_score": first(row, "raw_score", "raw", default="").strip(),
                "feature_version": first(row, "feature_version", default="").strip(),
                "resume_chars": first(row, "resume_chars", default="").strip(),
                # keep essays available for humans while labeling (optional)
                "essayMath": first(row, "essayMath", default=""),
                "essayCommunity": first(row, "essayCommunity", default=""),
            }

            obj = {"id": applicant_id, "features": feats, "meta": meta}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"wrote {n_ok} applicants to {out_path} (skipped {n_skip} rows with missing/bad features_json)")


if __name__ == "__main__":
    main()