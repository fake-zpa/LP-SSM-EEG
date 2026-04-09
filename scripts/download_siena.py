"""
Download and preprocess SIENA Scalp EEG Dataset (PhysioNet).

14 patients, scalp EEG, seizure annotations.
Resamples 512 Hz → 256 Hz, selects 22 common channels with CHB-MIT,
segments into 4-second non-overlapping windows, creates manifest CSV.

Usage:
    conda run -n mamba2 python scripts/download_siena.py [--subjects PN00,PN01] [--smoke]

Outputs:
    data/raw/siena/          (EDF files)
    data/processed/siena/    (npy segments)
    data/manifests/siena_manifest.csv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL     = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
RECORDS_URL  = f"{BASE_URL}/RECORDS"
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "siena"
PROC_DIR     = PROJECT_ROOT / "data" / "processed" / "siena"
MANIFEST_OUT = PROJECT_ROOT / "data" / "manifests" / "siena_manifest.csv"

TARGET_SFREQ = 256.0        # CHB-MIT uses 256 Hz
WINDOW_SEC   = 4.0          # 4-second windows (= 1024 samples at 256 Hz)
WINDOW_SAMP  = int(TARGET_SFREQ * WINDOW_SEC)  # 1024
STRIDE_SEC   = 2.0          # 2-second stride — matches CHB-MIT training preprocessing
STRIDE_SAMP  = int(TARGET_SFREQ * STRIDE_SEC)  # 512

N_CHANNELS_TARGET = 22   # CHB-MIT training data uses 22 channels

# Bipolar derivation table: (output_name, electrode_A, electrode_B)
# output = signal_A - signal_B
# SIENA uses old 10-20 naming: T3=T7, T4=T8, T5=P7, T6=P8
# FT9≈F9, FT10≈F10 in SIENA's electrode set
BIPOLAR_PAIRS = [
    ("FP1-F7",     "Fp1", "F7"),
    ("F7-T7",      "F7",  "T3"),
    ("T7-P7",      "T3",  "T5"),
    ("P7-O1",      "T5",  "O1"),
    ("FP1-F3",     "Fp1", "F3"),
    ("F3-C3",      "F3",  "C3"),
    ("C3-P3",      "C3",  "P3"),
    ("P3-O1",      "P3",  "O1"),
    ("FP2-F4",     "Fp2", "F4"),
    ("F4-C4",      "F4",  "C4"),
    ("C4-P4",      "C4",  "P4"),
    ("P4-O2",      "P4",  "O2"),
    ("FP2-F8",     "Fp2", "F8"),
    ("F8-T8",      "F8",  "T4"),
    ("T8-P8",      "T4",  "T6"),
    ("P8-O2",      "T6",  "O2"),
    ("FZ-CZ",      "Fz",  "Cz"),
    ("CZ-PZ",      "Cz",  "Pz"),
    ("P7-T7",      "T5",  "T3"),
    ("T7-FT9",     "T3",  "F9"),
    ("FT9-FT10",   "F9",  "F10"),
    ("FT10-T8",    "F10", "T4"),
]

# Bandpass filter matching CHB-MIT preprocessing (0.5–40 Hz)
BANDPASS = (0.5, 40.0)


def fetch_records():
    r = requests.get(RECORDS_URL, timeout=30)
    r.raise_for_status()
    return [l.strip() for l in r.text.splitlines() if l.strip().endswith(".edf")]


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"
        mode = "ab"
    else:
        mode = "wb"
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code in (200, 206):
                    with open(dest, mode) as f:
                        for chunk in resp.iter_content(65536):
                            f.write(chunk)
                    return True
                elif resp.status_code == 416:  # already complete
                    return True
                else:
                    print(f"  HTTP {resp.status_code}: {url}")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return False


def _hms_to_sec(hms_str: str) -> float | None:
    """Convert HH.MM.SS or HH:MM:SS to seconds. Handles embedded spaces (e.g. '1 6.49.25')."""
    import re
    # Strip all internal whitespace to fix typos like '1 6.49.25' → '16.49.25'
    hms_str = re.sub(r"\s+", "", hms_str.strip())
    m = re.match(r"(\d+)[.:]((\d+)[.:]?(\d+))", hms_str)
    if not m:
        return None
    try:
        parts = re.split(r"[.:]", hms_str)
        if len(parts) == 3:
            h, mn, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + mn * 60 + s
        elif len(parts) == 2:
            mn, s = int(parts[0]), int(parts[1])
            return mn * 60 + s
    except Exception:
        return None
    return None


def parse_siena_annotations(subj_dir: Path, edf_stem: str) -> list:
    """Parse SIENA seizure annotations from Seizures-list-{subj}.txt.

    Handles multiple SIENA format variants:
      - 'Seizure n1' / 'Seizure n 2' / 'Seizure n 1(in sleep):' etc.
      - 'Seizure start time:' OR 'Start time:'
      - 'Seizure end time:'   OR 'End time:'
      - File name typos (e.g. PNO6 vs PN06) — uses fuzzy stem match
      - Midnight wraparound (seizure time < registration time)
    Returns list of (start_sec, end_sec) relative to EDF start.
    """
    import re
    subj_id  = subj_dir.name
    ann_path = subj_dir / f"Seizures-list-{subj_id}.txt"
    if not ann_path.exists():
        return []

    seizures = []
    try:
        text = ann_path.read_text(errors="replace")
        # Split on seizure number markers (flexible pattern)
        blocks = re.split(r"Seizure\s+n\s*\d+", text, flags=re.IGNORECASE)
        for block in blocks[1:]:
            fn_m  = re.search(r"File name\s*:\s*([^\s\n]+\.edf)", block, re.IGNORECASE)
            reg_m = re.search(r"Registration start time\s*:\s*([\d.:]+)", block, re.IGNORECASE)
            # Match "Seizure start time:" OR standalone "Start time:" (not "Registration start")
            s_m = (re.search(r"Seizure\s+[Ss]tart\s+time\s*:\s*([\d.:]+)", block) or
                   re.search(r"(?m)^\s*[Ss]tart\s+time\s*:\s*([\d.:]+)", block))
            # Match "Seizure end time:" OR standalone "End time:" (not "Registration end")
            e_m = (re.search(r"Seizure\s+[Ee]nd\s+time\s*:\s*([\d.:]+)", block) or
                   re.search(r"(?m)^\s*[Ee]nd\s+time\s*:\s*([\d.:]+)", block))
            if not (fn_m and reg_m and s_m and e_m):
                continue

            # Fuzzy filename match: strip non-alphanum from both sides
            # Also allow prefix match: "PN01" matches "PN01-1"
            fn_clean   = re.sub(r"[^a-z0-9]", "", fn_m.group(1).lower().replace(".edf", ""))
            stem_clean = re.sub(r"[^a-z0-9]", "", edf_stem.lower())
            if fn_clean != stem_clean and not stem_clean.startswith(fn_clean):
                continue

            reg_sec  = _hms_to_sec(reg_m.group(1))
            sz_start = _hms_to_sec(s_m.group(1))
            sz_end   = _hms_to_sec(e_m.group(1))
            if None in (reg_sec, sz_start, sz_end):
                continue

            offset_s = sz_start - reg_sec
            offset_e = sz_end   - reg_sec
            # Midnight wraparound: seizure on following day
            if offset_s < -3600:
                offset_s += 86400
                offset_e += 86400
            if offset_e < offset_s:      # end before start → end wraps midnight
                offset_e += 86400
            if offset_e > offset_s >= 0:
                seizures.append((offset_s, offset_e))
    except Exception as e:
        print(f"  Ann parse error ({ann_path}): {e}")
    return seizures


def preprocess_edf(edf_path: Path, proc_dir: Path, subj_id: str) -> list:
    """Segment EDF into windows; return list of manifest rows."""
    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        print("  mne not available; skipping EDF preprocessing")
        return []

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"  EDF read error ({edf_path.name}): {e}")
        return []

    orig_sfreq = raw.info["sfreq"]
    n_channels  = len(raw.ch_names)

    # Resample if needed
    if abs(orig_sfreq - TARGET_SFREQ) > 1.0:
        raw.resample(TARGET_SFREQ, verbose=False)

    # Bandpass filter — matches CHB-MIT preprocessing pipeline
    raw.filter(BANDPASS[0], BANDPASS[1], method="fir", verbose=False)

    # Construct bipolar montage to match CHB-MIT training data
    # SIENA has unipolar channels; CHB-MIT uses bipolar derivations
    all_data = raw.get_data()  # [n_ch, n_times]
    ch_lower = {c.replace("EEG ", "").strip().lower(): i for i, c in enumerate(raw.ch_names)}

    bipolar_rows = []
    matched_names = []
    for bip_name, elec_a, elec_b in BIPOLAR_PAIRS:
        ia = ch_lower.get(elec_a.lower())
        ib = ch_lower.get(elec_b.lower())
        if ia is not None and ib is not None:
            bipolar_rows.append(all_data[ia] - all_data[ib])
            matched_names.append(bip_name)

    if len(bipolar_rows) < 10:
        print(f"    WARNING: only {len(bipolar_rows)} bipolar channels matched, "
              f"falling back to first {N_CHANNELS_TARGET} raw channels")
        picks = list(range(min(N_CHANNELS_TARGET, len(raw.ch_names))))
        data = raw.get_data(picks=picks)
    else:
        data = np.array(bipolar_rows)  # [C, T]

    # Pad or truncate to N_CHANNELS_TARGET
    actual_c = data.shape[0]
    if actual_c < N_CHANNELS_TARGET:
        pad = np.zeros((N_CHANNELS_TARGET - actual_c, data.shape[1]), dtype=data.dtype)
        data = np.vstack([data, pad])
    elif actual_c > N_CHANNELS_TARGET:
        data = data[:N_CHANNELS_TARGET]

    # Parse seizure annotations
    ann_dir  = edf_path.parent
    ann_stem = edf_path.stem
    seizures = parse_siena_annotations(ann_dir, ann_stem)

    # Segment into windows (2s stride, matching CHB-MIT training)
    total_samples = data.shape[1]
    starts = list(range(0, total_samples - WINDOW_SAMP + 1, STRIDE_SAMP))

    rows = []
    stem = edf_path.stem
    proc_dir.mkdir(parents=True, exist_ok=True)

    for w, start_s in enumerate(starts):
        end_s   = start_s + WINDOW_SAMP
        t_start = start_s / TARGET_SFREQ
        t_end   = end_s   / TARGET_SFREQ

        # Label: 1 if window overlaps with any seizure by >50%
        label = 0
        for sz_start, sz_end in seizures:
            overlap = max(0, min(t_end, sz_end) - max(t_start, sz_start))
            if overlap > (WINDOW_SEC * 0.5):
                label = 1
                break

        seg = data[:, start_s:end_s].astype(np.float32)
        # Per-channel z-score
        mu  = seg.mean(axis=1, keepdims=True)
        std = seg.std(axis=1, keepdims=True) + 1e-6
        seg = (seg - mu) / std

        npy_name = f"{stem}_w{w:05d}.npy"
        npy_path = proc_dir / subj_id / npy_name
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), seg)

        rows.append({
            "subject_id": subj_id,
            "edf":        str(edf_path.name),
            "window_idx": w,
            "t_start":    round(t_start, 3),
            "t_end":      round(t_end,   3),
            "label":      label,
            "npy_path":   str(npy_path.relative_to(PROJECT_ROOT)),
            "n_channels": N_CHANNELS_TARGET,
            "sfreq":      TARGET_SFREQ,
            "split":      "test",   # all SIENA windows used as external test
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default=None,
                        help="Comma-sep subject IDs, e.g. PN00,PN01")
    parser.add_argument("--smoke", action="store_true",
                        help="Only process PN00 (quick test)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only preprocess existing EDFs")
    args = parser.parse_args()

    # Filter by subject
    filter_subjs = None
    if args.subjects:
        filter_subjs = set(s.strip() for s in args.subjects.split(","))
    elif args.smoke:
        filter_subjs = {"PN00"}

    if args.skip_download:
        # Build record list from local files — avoids network call
        local_edfs = sorted(RAW_DIR.rglob("*.edf"))
        all_records = [str(e.relative_to(RAW_DIR)) for e in local_edfs]
        print(f"Skipping download. Found {len(all_records)} local EDF files.")
    else:
        print("Fetching SIENA RECORDS …")
        all_records = fetch_records()
        print(f"  {len(all_records)} EDF records found")

    if filter_subjs:
        all_records = [r for r in all_records
                       if any(r.startswith(s + "/") or r.startswith(s + "\\") for s in filter_subjs)]
    print(f"  Processing {len(all_records)} records" +
          (f" (filter: {filter_subjs})" if filter_subjs else ""))

    # Download EDFs + annotation files (parallel)
    if not args.skip_download:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_threads = 8 if not args.smoke else 2

        def download_one(rec):
            dest = RAW_DIR / rec
            if dest.exists() and dest.stat().st_size > 10000:
                return rec, True, "skip"
            url = f"{BASE_URL}/{rec}"
            ok = download_file(url, dest)
            return rec, ok, "ok" if ok else "FAILED"

        # Add subject-level annotation files (Seizures-list-{subj}.txt)
        all_subjects = sorted({Path(r).parent.name for r in all_records})
        ann_records  = [f"{s}/Seizures-list-{s}.txt" for s in all_subjects]
        all_to_download = list(all_records) + ann_records

        print(f"\nDownloading {len(all_to_download)} files ({n_threads} threads) …")
        done = 0
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = {ex.submit(download_one, rec): rec for rec in all_to_download}
            for fut in as_completed(futures):
                rec, ok, status = fut.result()
                done += 1
                if status != "skip":
                    print(f"  [{done}/{len(all_to_download)}] {Path(rec).name}: {status}", flush=True)

    # Preprocess
    print("\nPreprocessing EDFs …")
    all_rows = []
    for rec in all_records:
        edf_path = RAW_DIR / rec
        if not edf_path.exists():
            print(f"  [missing] {rec}")
            continue
        subj_id = Path(rec).parent.name   # e.g. PN00
        print(f"  Preprocessing {rec} …")
        rows = preprocess_edf(edf_path, PROC_DIR, subj_id)
        n_ictal = sum(r["label"] for r in rows)
        print(f"    → {len(rows)} windows, {n_ictal} ictal")
        all_rows.extend(rows)

    if not all_rows:
        print("No windows extracted!")
        return

    df = pd.DataFrame(all_rows)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(MANIFEST_OUT), index=False)
    print(f"\nManifest saved: {MANIFEST_OUT}")
    print(f"  Total: {len(df)} windows, {df['label'].sum()} ictal "
          f"({df['label'].mean()*100:.1f}%)")
    print(f"  Subjects: {sorted(df['subject_id'].unique())}")


if __name__ == "__main__":
    main()
