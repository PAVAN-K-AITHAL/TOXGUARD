import csv, os

datasets = [
    ("toxcast", "data/toxcast_final.csv"),
    ("tox21", "data/tox21_final.csv"),
    ("herg", "data/herg_final.csv"),
    ("dili", "data/dili_final.csv"),
    ("common_molecules", "data/common_molecules_final.csv"),
    ("t3db (external)", "data/t3db_processed.csv"),
]

total_tox, total_non = 0, 0

for name, path in datasets:
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tox, non = 0, 0
        for row in reader:
            val = row.get("is_toxic", "")
            if val in ("1", "1.0"):
                tox += 1
            elif val in ("0", "0.0"):
                non += 1
        total = tox + non
        ratio = tox / total * 100 if total else 0
        print(f"{name:20s}: {total:6d} total | {tox:6d} toxic ({ratio:5.1f}%) | {non:6d} non-toxic ({100-ratio:5.1f}%)")
        if "external" not in name:
            total_tox += tox
            total_non += non

print("-" * 75)
combined = total_tox + total_non
print(f"{'COMBINED (train)':20s}: {combined:6d} total | {total_tox:6d} toxic ({total_tox/combined*100:5.1f}%) | {total_non:6d} non-toxic ({total_non/combined*100:5.1f}%)")
