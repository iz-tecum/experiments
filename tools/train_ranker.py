#!/usr/bin/env python3
"""Train a simple pairwise ranker from (applicants.jsonl, pairs.csv).

Model: logistic regression on feature differences.
Given pair (i,j,y):
  y=1 means i should rank higher than j.
We minimize:
  sum log(1 + exp(-y' * (w·(xi-xj) + b))) + (lambda/2)||w||^2
where y' in {+1,-1}.

Outputs a JSON model you can drop into public/rank_model.json.
Schema:
{
  "feature_version": 2,
  "dim": 21,
  "w": [...],
  "b": 0.0
}
"""

import argparse, csv, json, math
from pathlib import Path


def load_applicants_jsonl(path: Path):
    id2x = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            id_ = str(obj.get("id"))
            x = obj.get("features")
            if not id_ or not isinstance(x, list):
                continue
            id2x[id_] = [float(v) for v in x]
    return id2x


def load_pairs_csv(path: Path):
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            i = str(r.get("i", ""))
            j = str(r.get("j", ""))
            y = r.get("y", "")
            if not i or not j:
                continue
            try:
                y = int(float(y))
            except Exception:
                continue
            if y not in (0, 1):
                continue
            pairs.append((i, j, y))
    return pairs


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def train(id2x, pairs, dim, lr, epochs, l2):
    w = [0.0] * dim
    b = 0.0

    for ep in range(1, epochs + 1):
        loss = 0.0
        n = 0
        for i, j, y01 in pairs:
            xi = id2x.get(i)
            xj = id2x.get(j)
            if xi is None or xj is None:
                continue

            y = 1.0 if y01 == 1 else -1.0
            dx = [a - b_ for a, b_ in zip(xi, xj)]
            s = dot(w, dx) + b
            z = y * s

            if z > 35:
                ell = 0.0
                sig = 0.0
            elif z < -35:
                ell = -z
                sig = 1.0
            else:
                ell = math.log1p(math.exp(-z))
                sig = 1.0 / (1.0 + math.exp(z))  # sigmoid(-z)

            loss += ell
            n += 1

            gscale = -y * sig
            for k in range(dim):
                w[k] -= lr * (gscale * dx[k] + l2 * w[k])
            b -= lr * (gscale)

        if n == 0:
            raise SystemExit("No usable pairs (IDs not found in applicants.jsonl?).")

        l2term = 0.5 * l2 * sum(v*v for v in w)
        print(f"epoch {ep:03d}: avg_loss={(loss/n):.6f}  l2term={l2term:.6f}  used_pairs={n}")

    return w, b


def save_model(path: Path, feature_version: int, w, b):
    obj = {
        "feature_version": feature_version,
        "dim": len(w),
        "weights": [round(float(v), 12) for v in w],
        "bias": round(float(b), 12),
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--applicants", required=True, help="Path to applicants.jsonl")
    ap.add_argument("--pairs", required=True, help="Path to pairs.csv")
    ap.add_argument("--out", default="rank_model.json", help="Output model json")
    ap.add_argument("--feature-version", type=int, default=2)
    ap.add_argument("--dim", type=int, default=21)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--l2", type=float, default=0.001)
    args = ap.parse_args()

    id2x = load_applicants_jsonl(Path(args.applicants))
    pairs = load_pairs_csv(Path(args.pairs))
    print(f"Loaded {len(id2x)} applicants, {len(pairs)} labeled pairs")

    w, b = train(id2x, pairs, args.dim, args.lr, args.epochs, args.l2)
    save_model(Path(args.out), args.feature_version, w, b)
    print(f"Wrote model -> {args.out}")


if __name__ == "__main__":
    main()