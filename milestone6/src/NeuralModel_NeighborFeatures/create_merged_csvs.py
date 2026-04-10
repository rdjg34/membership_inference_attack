"""
create_merged_csvs.py
=====================
Merges the teammate's 3 family CSVs (1_temperature, 2_renyi, 3_img_pert) with
the *_neighbor_only.csv files to produce train/val/test_merged.csv in this directory.

Run this after extract_features.py (or after copying neighbor CSVs from milestone6).
The merged CSVs are saved here for inspection; the main pipeline reads the source
CSVs directly and does this merge in memory.

Usage:
    python create_merged_csvs.py
    python create_merged_csvs.py --data-root /path/to/extracted_features
    python create_merged_csvs.py --splits train val
"""

#Help from Claude


import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE.parents[1] / "data" / "extracted_features"

FAMILIES = {
    "temp":  "1_temperature",
    "renyi": "2_renyi",
    "img":   "3_img_pert",
}


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return cols, rows


def create_merged(data_root: Path, src_dir: Path, split: str) -> None:
    # Load and merge all 3 family CSVs (matching lightweight_pipeline.py merge logic)
    family_rows_by_id: dict[str, dict] = {}
    all_family_cols: list[str] = []
    label_by_id: dict[str, str] = {}

    for fkey, fname in FAMILIES.items():
        fam_path = data_root / fname / f"{split}.csv"
        cols, rows = read_csv(fam_path)
        for row in rows:
            sid = row["id"]
            if sid not in family_rows_by_id:
                family_rows_by_id[sid] = {"id": sid}
                if "label" in row:
                    label_by_id[sid] = row["label"]
            for c in cols:
                if c not in {"id", "label"}:
                    family_rows_by_id[sid][f"{fkey}__{c}"] = row[c]
                    col_key = f"{fkey}__{c}"
                    if col_key not in all_family_cols:
                        all_family_cols.append(col_key)

    # Load neighbor features
    nb_path = src_dir / f"{split}_neighbor_only.csv"
    nb_cols, nb_rows = read_csv(nb_path)
    nb_by_id = {row["id"]: row for row in nb_rows}

    # Neighbor cols to add (exclude id/label — those come from family CSVs)
    nb_extra_cols = [c for c in nb_cols if c not in {"id", "label", "is_member"}]
    nb_prefixed = [f"nb__{c}" for c in nb_extra_cols]

    # Build merged rows (inner join on ids present in all sources)
    ids = [sid for sid in family_rows_by_id if sid in nb_by_id]
    merged_cols = ["id"] + all_family_cols + (["label"] if label_by_id else []) + nb_prefixed
    merged_rows = []
    for sid in ids:
        row = dict(family_rows_by_id[sid])
        if label_by_id:
            row["label"] = label_by_id.get(sid, "")
        nb = nb_by_id[sid]
        for c, pc in zip(nb_extra_cols, nb_prefixed):
            row[pc] = nb[c]
        merged_rows.append(row)

    out_path = src_dir / f"{split}_merged.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_cols)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved → {out_path}  ({len(merged_rows)} rows, {len(merged_cols)} cols)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root dir containing 1_temperature/, 2_renyi/, 3_img_pert/"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    src_dir   = HERE

    print(f"Data root : {data_root}")
    print(f"Output dir: {src_dir}")

    for split in args.splits:
        print(f"\nMerging {split}...")
        create_merged(data_root, src_dir, split)


if __name__ == "__main__":
    main()
