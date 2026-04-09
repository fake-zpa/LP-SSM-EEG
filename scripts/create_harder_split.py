"""
Create a harder patient-wise split from existing manifest for generalization experiment.

Harder split (E09):
  Train: chb01, chb02, chb06    (3 subjects, 154 ictal)
  Val:   chb03                  (1 subject,  206 ictal)
  Test:  chb04, chb05, chb10    (3 subjects, 532 ictal)

vs Primary split (E01-E07):
  Train: chb01, chb02, chb03, chb06
  Val:   chb05
  Test:  chb04, chb10

This creates a harder generalization test: fewer training subjects + 3 unseen test patients.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

src_manifest = PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv"
dst_manifest = PROJECT_ROOT / "data" / "manifests" / "chbmit_harder_split.csv"

TRAIN_SUBJECTS = {"chb01", "chb02", "chb06"}
VAL_SUBJECTS   = {"chb03"}
TEST_SUBJECTS  = {"chb04", "chb05", "chb10"}

df = pd.read_csv(src_manifest)

def assign_split(row):
    s = row["subject_id"]
    if s in TRAIN_SUBJECTS:
        return "train"
    elif s in VAL_SUBJECTS:
        return "val"
    elif s in TEST_SUBJECTS:
        return "test"
    else:
        return "excluded"

df["split"] = df.apply(assign_split, axis=1)

dst_manifest.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(dst_manifest, index=False)

print(f"Harder-split manifest saved: {dst_manifest}")
print(f"Total windows: {len(df)}")
print()
for split in ["train", "val", "test", "excluded"]:
    sub = df[df["split"] == split]
    subjects = sorted(sub["subject_id"].unique())
    n_ictal = (sub["label"] == 1).sum()
    n_total = len(sub)
    print(f"  {split:10s}: {subjects} → {n_ictal} ictal / {n_total} total")
