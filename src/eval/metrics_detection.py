"""Event-level detection metrics for CHB-MIT seizure detection."""
import numpy as np
from typing import List, Tuple, Dict


def compute_event_metrics(
    pred_probs: np.ndarray,
    true_labels: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.5,
    min_event_dur_sec: float = 2.0,
    merge_gap_sec: float = 5.0,
    recording_duration_hours: float = 1.0,
) -> Dict[str, float]:
    """
    Compute event-level metrics:
      - event_sensitivity: fraction of true seizure events detected
      - false_alarms_per_hour: number of false positive events per hour
      - mean_onset_latency_sec: mean seconds between true onset and first detection
    """
    preds = (pred_probs >= threshold).astype(int)

    true_events = _binary_to_events(true_labels, times, min_dur=min_event_dur_sec, gap=merge_gap_sec)
    pred_events = _binary_to_events(preds, times, min_dur=min_event_dur_sec, gap=merge_gap_sec)

    if len(true_events) == 0:
        return {
            "event_sensitivity": float("nan"),
            "false_alarms_per_hour": len(pred_events) / max(recording_duration_hours, 1),
            "mean_onset_latency_sec": float("nan"),
        }

    detected = 0
    latencies = []
    matched_preds = set()

    for t_on, t_off in true_events:
        for i, (p_on, p_off) in enumerate(pred_events):
            if i in matched_preds:
                continue
            if p_on <= t_off and p_off >= t_on:
                detected += 1
                latencies.append(max(0.0, p_on - t_on))
                matched_preds.add(i)
                break

    fp_events = len(pred_events) - len(matched_preds)

    return {
        "event_sensitivity": detected / len(true_events),
        "false_alarms_per_hour": fp_events / max(recording_duration_hours, 1),
        "mean_onset_latency_sec": float(np.mean(latencies)) if latencies else float("nan"),
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
        "n_detected": detected,
    }


def _binary_to_events(
    labels: np.ndarray,
    times: np.ndarray,
    min_dur: float = 2.0,
    gap: float = 5.0,
) -> List[Tuple[float, float]]:
    """Convert binary label array to list of (onset, offset) event tuples."""
    events = []
    in_event = False
    start = None
    last_end = None

    for i, (t, l) in enumerate(zip(times, labels)):
        if l == 1 and not in_event:
            if last_end is not None and (t - last_end) <= gap:
                start = events[-1][0] if events else t
                events = events[:-1] if events else events
            else:
                start = t
            in_event = True
        elif l == 0 and in_event:
            dur = t - start
            if dur >= min_dur:
                events.append((start, t))
                last_end = t
            in_event = False

    if in_event and times[-1] - start >= min_dur:
        events.append((start, times[-1]))

    return events
