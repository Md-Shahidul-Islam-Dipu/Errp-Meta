"""INRIA NER 2015 loader + preprocessing (verbatim) + preprocessing-depth variants.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from ..progress import tqdm

from ..config import Config

def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract (subject_id, session_num) from 'Data_S02_Sess01.csv'."""
    m = re.match(r'Data_S(\d+)_Sess(\d+)\.csv', filename)
    return (f"S{m.group(1)}", int(m.group(2))) if m else (None, None)

def index_dataset(data_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """Return {subject_id: [(session_num, filepath), ...]} sorted by session."""
    idx: Dict = defaultdict(list)
    for fp in glob.glob(os.path.join(data_dir, "*.csv")):
        sid, sess = parse_filename(os.path.basename(fp))
        if sid:
            idx[sid].append((sess, fp))
    for k in idx:
        idx[k].sort()
    return dict(idx)

def load_labels(labels_file: str) -> Dict[str, Dict[int, List[Tuple[int, int]]]]:
    """Parse TrainLabels.csv → {subject → {session → [(fb_idx, label)]}}."""
    df = pd.read_csv(labels_file)
    ld: Dict = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        m = re.match(r'S(\d+)_Sess(\d+)_FB(\d+)', str(row['IdFeedBack']))
        if not m:
            continue
        sid  = f"S{m.group(1)}"
        sess = int(m.group(2))
        fbi  = int(m.group(3))
        ld[sid][sess].append((fbi, int(row['Prediction'])))
    for sid in ld:
        for sess in ld[sid]:
            ld[sid][sess].sort(key=lambda x: x[0])
    return {k: dict(v) for k, v in ld.items()}

def build_dataset_mapping(train_index: Dict, labels_dict: Dict) -> Dict[str, List[Dict]]:
    """Join file index with label lists into a single mapping."""
    mapping = {}
    for sid, sessions in train_index.items():
        if sid not in labels_dict:
            continue
        mapping[sid] = []
        for sess, fp in sessions:
            if sess not in labels_dict[sid]:
                continue
            mapping[sid].append({'session_num': sess, 'filepath': fp,
                                  'labels': labels_dict[sid][sess]})
    return mapping

def load_continuous_eeg(filepath: str) -> Tuple[pd.DataFrame, float]:
    """Load CSV EEG file. Infer sampling frequency from Time column."""
    df = pd.read_csv(filepath)
    dt = float(np.median(np.diff(df['Time'].values[:200])))
    return df, 1.0 / dt

def get_eeg_channels(df: pd.DataFrame) -> List[str]:
    """Return column names excluding Time and FeedBackEvent."""
    return [c for c in df.columns if c not in ('Time', 'FeedBackEvent')]

def apply_filters_continuous(eeg_data: np.ndarray, sfreq: float,
                              lowcut: float, highcut: float,
                              notch_freq: float, order: int = 4) -> np.ndarray:
    """Apply notch then bandpass to continuous EEG (n_channels × n_samples).

    Uses zero-phase sosfiltfilt to avoid phase distortion.
    Applied BEFORE epoching to prevent boundary edge artifacts.

    Args:
        eeg_data  : Shape (n_channels, n_samples), raw µV.
        sfreq     : Sampling frequency in Hz.
        lowcut    : Bandpass lower cutoff (Hz).
        highcut   : Bandpass upper cutoff (Hz).
        notch_freq: Power-line notch (Hz).
        order     : Butterworth filter order.
    """
    nyq = sfreq / 2.0
    out = eeg_data.copy()

    # 1. Notch filter
    nl, nh = (notch_freq - 1.0) / nyq, (notch_freq + 1.0) / nyq
    if 0 < nl < 1 and 0 < nh < 1:
        sos = sp_signal.butter(order, [nl, nh], btype='bandstop', output='sos')
        out = sp_signal.sosfiltfilt(sos, out, axis=1)

    # 2. Bandpass filter
    lo, hi = lowcut / nyq, min(highcut / nyq, 0.9999)
    if 0 < lo < hi < 1:
        sos = sp_signal.butter(order, [lo, hi], btype='band', output='sos')
        out = sp_signal.sosfiltfilt(sos, out, axis=1)

    return out

def detect_feedback_events(df: pd.DataFrame) -> np.ndarray:
    """Return sample indices of 0→1 transitions in FeedBackEvent column."""
    ev = df['FeedBackEvent'].values
    return np.where(np.diff(np.concatenate([[0], ev])) == 1)[0]

def create_epochs(eeg_data: np.ndarray, event_indices: np.ndarray,
                  sfreq: float, tmin: float, tmax: float
                  ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Slice epochs from filtered continuous EEG.

    Returns:
        epochs_data     : (n_valid, n_ch, n_times)
        times           : (n_times,) time axis in seconds
        valid_positions : list mapping epoch index → original event position
    """
    n_before = int(abs(tmin) * sfreq)
    n_after  = int(tmax * sfreq)
    n_total  = eeg_data.shape[1]
    epochs, valid_pos = [], []
    for pos, idx in enumerate(event_indices):
        s, e = idx - n_before, idx + n_after
        if s >= 0 and e <= n_total:
            epochs.append(eeg_data[:, s:e])
            valid_pos.append(pos)
    arr   = np.array(epochs) if epochs else np.empty((0, eeg_data.shape[0], n_before + n_after))
    times = np.linspace(tmin, tmax, n_before + n_after, endpoint=False)
    return arr, times, valid_pos

def align_epochs_with_labels_by_index(
        epochs_data: np.ndarray,
        valid_positions: List[int],
        session_labels: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Align epochs to labels using the feedback index — NOT positional truncation.

    Positional truncation (min(n_epochs, n_labels)) is WRONG:
    if any epoch is dropped at a boundary, all subsequent labels shift silently.

    This function matches epoch i to the label whose event position equals
    valid_positions[i], using the ordered feedback list as the position reference.
    """
    pos_to_label = {pos: lbl for pos, (_, lbl) in enumerate(session_labels)}
    aligned_e, aligned_l = [], []
    for ep_i, orig_pos in enumerate(valid_positions):
        if orig_pos in pos_to_label:
            aligned_e.append(epochs_data[ep_i])
            aligned_l.append(pos_to_label[orig_pos])
    if not aligned_e:
        return np.empty((0,) + epochs_data.shape[1:]), np.empty(0, dtype=int)
    return np.array(aligned_e), np.array(aligned_l, dtype=int)

def reject_artifacts(epochs_data: np.ndarray,
                     threshold_uv: float = 100.0) -> np.ndarray:
    """Return boolean mask (True = keep) for epochs below peak-to-peak threshold.

    Applied on RAW µV BEFORE baseline correction or z-score.
    The 100 µV threshold is standard in ERP literature (Luck 2014).
    """
    ptp = np.ptp(epochs_data, axis=2).max(axis=1)   # max ptp across channels
    return ptp <= threshold_uv

def baseline_correct(epochs_data: np.ndarray, times: np.ndarray,
                     baseline: Tuple[float, float] = (-0.2, 0.0)) -> np.ndarray:
    """Subtract per-epoch per-channel mean of the baseline window.

    Applied AFTER artifact rejection on CLEAN epochs.
    Computed independently for each epoch and each channel.
    """
    t0, t1 = baseline
    mask = (times >= t0) & (times <= t1)
    assert mask.sum() > 0, f"Baseline window {baseline} is empty (times range: {times[[0,-1]]})"
    bl_mean = epochs_data[:, :, mask].mean(axis=2, keepdims=True)  # (N, C, 1)
    return epochs_data - bl_mean

def process_subject_data(subject_id: str, session_infos: List[Dict]) -> Optional[Dict]:
    """Complete preprocessing pipeline for one subject across all sessions.

    Pipeline:
        continuous load → continuous filter → epoch → index-align →
        artifact reject (µV) → baseline correct → concatenate sessions

    Returns dict: epochs (N,C,T), labels (N,), ch_names, times, sfreq,
                  n_rejected, n_raw.
    """
    all_epochs, all_labels = [], []
    sfreqs: List[float] = []
    ch_names_ref: Optional[List[str]] = None
    times_ref: Optional[np.ndarray] = None
    total_raw = total_rejected = 0

    for sess_info in session_infos:
        try:
            df, sfreq = load_continuous_eeg(sess_info['filepath'])
            ch_names  = get_eeg_channels(df)
            raw_data  = df[ch_names].values.T  # (n_ch, n_samples)

            # Step 1: filter continuous signal
            raw_filt = apply_filters_continuous(
                raw_data, sfreq,
                Config.LOWCUT, Config.HIGHCUT, Config.NOTCH_FREQ, Config.FILTER_ORDER)

            # Step 2: detect events + create epochs
            events = detect_feedback_events(df)
            epo, times, valid_pos = create_epochs(
                raw_filt, events, sfreq, Config.TMIN, Config.TMAX)
            if len(epo) == 0:
                continue

            # Step 3: align by feedback index
            epo_al, lbl_al = align_epochs_with_labels_by_index(
                epo, valid_pos, sess_info['labels'])
            if len(epo_al) == 0:
                continue

            # Step 4: artifact rejection on raw µV
            mask = reject_artifacts(epo_al, Config.ART_THRESHOLD_UV)
            n_rej = int((~mask).sum())
            total_raw      += len(epo_al)
            total_rejected += n_rej
            epo_clean  = epo_al[mask]
            lbl_clean  = lbl_al[mask]
            if len(epo_clean) == 0:
                continue

            # Step 5: baseline correction on clean epochs
            epo_bc = baseline_correct(epo_clean, times, Config.BASELINE)

            all_epochs.append(epo_bc)
            all_labels.append(lbl_clean)
            sfreqs.append(sfreq)
            ch_names_ref = ch_names
            times_ref    = times

        except Exception as exc:
            print(f"  [{subject_id}] session {sess_info.get('session_num','?')} failed: {exc}")

    if not all_epochs:
        return None

    epochs_cat = np.concatenate(all_epochs, axis=0).astype(np.float32)
    labels_cat = np.concatenate(all_labels, axis=0).astype(int)
    sfreq_mean = float(np.mean(sfreqs))
    rej_rate   = 100.0 * total_rejected / max(total_raw, 1)
    cc = np.bincount(labels_cat, minlength=2)
    print(f"  {subject_id}: {len(epochs_cat)} epochs "
          f"(rej {total_rejected}/{total_raw}, {rej_rate:.1f}%) "
          f"| cls0={cc[0]} cls1={cc[1]}")

    return {'epochs': epochs_cat, 'labels': labels_cat,
            'ch_names': ch_names_ref, 'times': times_ref,
            'sfreq': sfreq_mean, 'n_sessions': len(session_infos),
            'n_rejected': total_rejected, 'n_raw': total_raw}

def process_subject_data_full(subject_id: str,
                               session_infos: List[Dict]) -> Optional[Dict]:
    """
    FULL pipeline:
      continuous filter (bandpass + notch)
      → epoch → index-align
      → artifact rejection (100 µV)
      → baseline correction
    """
    all_epochs, all_labels = [], []
    sfreqs: List[float] = []
    ch_names_ref = times_ref = None
    total_raw = total_rejected = 0

    for sess_info in session_infos:
        try:
            df, sfreq = load_continuous_eeg(sess_info['filepath'])
            ch_names  = get_eeg_channels(df)
            raw_data  = df[ch_names].values.T

            # ── STEP 1: filter continuous signal ──
            raw_filt = apply_filters_continuous(
                raw_data, sfreq,
                Config.LOWCUT, Config.HIGHCUT, Config.NOTCH_FREQ, Config.FILTER_ORDER)

            # ── STEP 2: epoch ──
            events = detect_feedback_events(df)
            epo, times, valid_pos = create_epochs(
                raw_filt, events, sfreq, Config.TMIN, Config.TMAX)
            if len(epo) == 0:
                continue

            # ── STEP 3: index-align ──
            epo_al, lbl_al = align_epochs_with_labels_by_index(
                epo, valid_pos, sess_info['labels'])
            if len(epo_al) == 0:
                continue

            # ── STEP 4: artifact rejection ──
            mask = reject_artifacts(epo_al, Config.ART_THRESHOLD_UV)
            total_raw      += len(epo_al)
            total_rejected += int((~mask).sum())
            epo_clean = epo_al[mask]
            lbl_clean = lbl_al[mask]
            if len(epo_clean) == 0:
                continue

            # ── STEP 5: baseline correction ──
            epo_bc = baseline_correct(epo_clean, times, Config.BASELINE)

            all_epochs.append(epo_bc)
            all_labels.append(lbl_clean)
            sfreqs.append(sfreq)
            ch_names_ref = ch_names
            times_ref    = times

        except Exception as exc:
            print(f"  [{subject_id}] session {sess_info.get('session_num','?')} failed: {exc}")

    if not all_epochs:
        return None

    epochs_cat = np.concatenate(all_epochs, axis=0).astype(np.float32)
    labels_cat = np.concatenate(all_labels, axis=0).astype(int)
    rej_rate   = 100.0 * total_rejected / max(total_raw, 1)
    cc = np.bincount(labels_cat, minlength=2)
    print(f"  {subject_id}: {len(epochs_cat)} epochs "
          f"(rej {total_rejected}/{total_raw}, {rej_rate:.1f}%) "
          f"| cls0={cc[0]} cls1={cc[1]}")

    return {'epochs': epochs_cat, 'labels': labels_cat,
            'ch_names': ch_names_ref, 'times': times_ref,
            'sfreq': float(np.mean(sfreqs)),
            'n_rejected': total_rejected, 'n_raw': total_raw,
            'condition': 'Full'}

def process_subject_data_filter_only(subject_id: str,
                                      session_infos: List[Dict]) -> Optional[Dict]:
    """
    FILTER ONLY pipeline:
      continuous filter (bandpass + notch)   ← active
      → epoch → index-align
      [artifact rejection]                   ← SKIPPED
      [baseline correction]                  ← SKIPPED

    Note: apply_filters_continuous is NOT modified; it is called normally.
    Artifact rejection and baseline correction are simply not called.
    """
    all_epochs, all_labels = [], []
    sfreqs: List[float] = []
    ch_names_ref = times_ref = None
    total_raw = 0  # no rejection in this condition

    for sess_info in session_infos:
        try:
            df, sfreq = load_continuous_eeg(sess_info['filepath'])
            ch_names  = get_eeg_channels(df)
            raw_data  = df[ch_names].values.T

            # ── STEP 1: filter continuous signal ── (ACTIVE)
            raw_filt = apply_filters_continuous(
                raw_data, sfreq,
                Config.LOWCUT, Config.HIGHCUT, Config.NOTCH_FREQ, Config.FILTER_ORDER)

            # ── STEP 2: epoch ──
            events = detect_feedback_events(df)
            epo, times, valid_pos = create_epochs(
                raw_filt, events, sfreq, Config.TMIN, Config.TMAX)
            if len(epo) == 0:
                continue

            # ── STEP 3: index-align ──
            epo_al, lbl_al = align_epochs_with_labels_by_index(
                epo, valid_pos, sess_info['labels'])
            if len(epo_al) == 0:
                continue

            # ── STEP 4: artifact rejection ── SKIPPED
            # reject_artifacts(...) not called — all epochs retained
            total_raw += len(epo_al)

            # ── STEP 5: baseline correction ── SKIPPED
            # baseline_correct(...) not called — raw filtered signal used as-is

            all_epochs.append(epo_al)    # unrejected, uncorrected
            all_labels.append(lbl_al)
            sfreqs.append(sfreq)
            ch_names_ref = ch_names
            times_ref    = times

        except Exception as exc:
            print(f"  [{subject_id}] session {sess_info.get('session_num','?')} failed: {exc}")

    if not all_epochs:
        return None

    epochs_cat = np.concatenate(all_epochs, axis=0).astype(np.float32)
    labels_cat = np.concatenate(all_labels, axis=0).astype(int)
    cc = np.bincount(labels_cat, minlength=2)
    print(f"  {subject_id}: {len(epochs_cat)} epochs "
          f"(no rejection) | cls0={cc[0]} cls1={cc[1]}")

    return {'epochs': epochs_cat, 'labels': labels_cat,
            'ch_names': ch_names_ref, 'times': times_ref,
            'sfreq': float(np.mean(sfreqs)),
            'n_rejected': 0, 'n_raw': total_raw,
            'condition': 'FilterOnly'}

def process_subject_data_artifact_baseline_only(subject_id: str,
                                                 session_infos: List[Dict]) -> Optional[Dict]:
    """
    ARTIFACT + BASELINE ONLY pipeline:
      [continuous filter]                    ← SKIPPED (apply_filters_continuous not called)
      → epoch (from RAW unfiltered signal)
      → index-align
      artifact rejection (100 µV)            ← active
      baseline correction                    ← active

    Note: apply_filters_continuous is NOT modified — it just isn't called here.
    Epoching is performed directly on the raw µV signal.
    """
    all_epochs, all_labels = [], []
    sfreqs: List[float] = []
    ch_names_ref = times_ref = None
    total_raw = total_rejected = 0

    for sess_info in session_infos:
        try:
            df, sfreq = load_continuous_eeg(sess_info['filepath'])
            ch_names  = get_eeg_channels(df)
            raw_data  = df[ch_names].values.T

            # ── STEP 1: continuous filter ── SKIPPED
            # apply_filters_continuous(...) not called — epoch raw signal directly
            raw_for_epoching = raw_data   # unfiltered

            # ── STEP 2: epoch ──
            events = detect_feedback_events(df)
            epo, times, valid_pos = create_epochs(
                raw_for_epoching, events, sfreq, Config.TMIN, Config.TMAX)
            if len(epo) == 0:
                continue

            # ── STEP 3: index-align ──
            epo_al, lbl_al = align_epochs_with_labels_by_index(
                epo, valid_pos, sess_info['labels'])
            if len(epo_al) == 0:
                continue

            # ── STEP 4: artifact rejection ── (ACTIVE — on raw µV as required)
            mask = reject_artifacts(epo_al, Config.ART_THRESHOLD_UV)
            total_raw      += len(epo_al)
            total_rejected += int((~mask).sum())
            epo_clean = epo_al[mask]
            lbl_clean = lbl_al[mask]
            if len(epo_clean) == 0:
                continue

            # ── STEP 5: baseline correction ── (ACTIVE)
            epo_bc = baseline_correct(epo_clean, times, Config.BASELINE)

            all_epochs.append(epo_bc)
            all_labels.append(lbl_clean)
            sfreqs.append(sfreq)
            ch_names_ref = ch_names
            times_ref    = times

        except Exception as exc:
            print(f"  [{subject_id}] session {sess_info.get('session_num','?')} failed: {exc}")

    if not all_epochs:
        return None

    epochs_cat = np.concatenate(all_epochs, axis=0).astype(np.float32)
    labels_cat = np.concatenate(all_labels, axis=0).astype(int)
    rej_rate   = 100.0 * total_rejected / max(total_raw, 1)
    cc = np.bincount(labels_cat, minlength=2)
    print(f"  {subject_id}: {len(epochs_cat)} epochs "
          f"(rej {total_rejected}/{total_raw}, {rej_rate:.1f}%) "
          f"| cls0={cc[0]} cls1={cc[1]}")

    return {'epochs': epochs_cat, 'labels': labels_cat,
            'ch_names': ch_names_ref, 'times': times_ref,
            'sfreq': float(np.mean(sfreqs)),
            'n_rejected': total_rejected, 'n_raw': total_raw,
            'condition': 'ArtifactBaselineOnly'}

PREPROC_FN = {
    'Full'               : process_subject_data_full,
    'FilterOnly'         : process_subject_data_filter_only,
    'ArtifactBaselineOnly': process_subject_data_artifact_baseline_only,
}
