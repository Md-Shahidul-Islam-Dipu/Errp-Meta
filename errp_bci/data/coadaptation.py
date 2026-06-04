"""ErrP-Coadaptation (Ehrlich & Cheng) loader: EEGLAB .set + ErrP event auto-detection.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
import glob
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
import mne
from scipy import signal as sp_signal
from ..progress import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..config import Config

def _resolve_coadaptation_dir(default="/kaggle/input/datasets/dipuislam/errp-coadaption/dataset-ErrP-coadaptation/data_coadaptation"):
    candidates = [default]
    for slug in ["dataset-errp-coadaptation", "errp-coadaptation",
                 "errp-co-adaptation", "ehrlich-errp-coadaptation"]:
        candidates += glob.glob(f"/kaggle/input/**/*{slug}*/data_coadaptation", recursive=True)
        candidates += glob.glob(f"/kaggle/input/**/*{slug}*", recursive=True)
    for p in candidates:
        if os.path.isdir(p) and any(f.startswith("s") for f in os.listdir(p)):
            return p
    return default

_FEEDBACK_COUNT_BOUNDS = (30, 600)

_MIDLINE_PREF = ['FCz', 'Cz', 'FC1', 'FC2', 'Fz', 'CPz']

_DIAG_TMIN, _DIAG_TMAX = -0.2, 1.0

def _pick_midline_channel(ch_names):
    name_set = {c.lower(): c for c in ch_names}
    for pref in _MIDLINE_PREF:
        if pref.lower() in name_set:
            return name_set[pref.lower()]
    for c in ch_names:
        cl = c.lower()
        if cl.startswith('fc') or cl in ('cz', 'fz', 'cpz'):
            return c
    return ch_names[0] if ch_names else None

def _grand_average(set_filepath, event_type, channel):
    raw = mne.io.read_raw_eeglab(set_filepath, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    raw.filter(1.0, 40.0, picks='eeg', verbose=False)
    if channel not in raw.ch_names:
        return None
    ch_idx = raw.ch_names.index(channel)
    data = raw.get_data()[ch_idx] * 1e6
    n_before = int(abs(_DIAG_TMIN) * sfreq)
    n_after  = int(_DIAG_TMAX * sfreq)
    first_samp = int(raw.first_samp)
    epochs = []
    for annot in raw.annotations:
        if annot['description'] != event_type:
            continue
        center = int(round(annot['onset'] * sfreq)) - first_samp
        s, e = center - n_before, center + n_after
        if 0 <= s and e <= data.shape[0]:
            seg = data[s:e].copy()
            seg -= seg[:n_before].mean()
            epochs.append(seg)
    if not epochs:
        return None
    arr = np.array(epochs)
    times = np.linspace(_DIAG_TMIN, _DIAG_TMAX, arr.shape[1], endpoint=False)
    return times, arr.mean(axis=0), arr.shape[0]

def _ne_pe_score(times, mean_uV):
    ne_mask = (times >= 0.20) & (times <= 0.35)
    pe_mask = (times >= 0.30) & (times <= 0.50)
    if not ne_mask.any() or not pe_mask.any():
        return 0.0
    return float(mean_uV[pe_mask].max() - mean_uV[ne_mask].min())

def autodetect_event_map(data_dir, count_bounds=_FEEDBACK_COUNT_BOUNDS,
                         save_diag_path='./diagnostics_coadapt'):
    os.makedirs(save_diag_path, exist_ok=True)
    set_files = sorted(glob.glob(os.path.join(data_dir, 's*', 's*_*.set')))
    if not set_files:
        raise FileNotFoundError(f'No .set files under {data_dir}')
    print(f'Scanning {len(set_files)} .set files for event types...')

    per_event = defaultdict(list)
    for fp in set_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                raw = mne.io.read_raw_eeglab(fp, preload=False, verbose=False)
        except Exception:
            continue
        counts = Counter(a['description'] for a in raw.annotations)
        for ev, n in counts.items():
            per_event[ev].append(n)
    if not per_event:
        raise RuntimeError('No events found in any .set file.')

    summary = [{'event': ev, 'n_sessions': len(c),
                'median_count': int(np.median(c)),
                'plausible_feedback': bool(count_bounds[0] <= np.median(c) <= count_bounds[1])}
               for ev, c in per_event.items()]
    summary_df = pd.DataFrame(summary).sort_values(
        ['plausible_feedback', 'median_count'], ascending=[False, True])
    summary_df.to_csv(f'{save_diag_path}/event_summary.csv', index=False)
    print('=== Event-type summary ===')
    print(summary_df.to_string(index=False))

    candidates = summary_df[summary_df['plausible_feedback']]['event'].tolist()
    if not candidates:
        raise RuntimeError(
            f'No event types have feedback-plausible counts in {count_bounds}. '
            'Inspect event_summary.csv and override via MANUAL_ERRP_EVENT_LABEL_MAP.')

    probe = set_files[0]
    raw = mne.io.read_raw_eeglab(probe, preload=False, verbose=False)
    channel = _pick_midline_channel(raw.ch_names)
    print(f'\nProbe file : {os.path.basename(probe)}')
    print(f'Channels   : {raw.ch_names}')
    print(f'Midline ch : {channel}')

    fig, axes = plt.subplots(len(candidates), 1,
                             figsize=(9, 2.2 * len(candidates)),
                             sharex=True, squeeze=False)
    scores = []
    for ax, ev in zip(axes[:, 0], candidates):
        result = _grand_average(probe, ev, channel)
        if result is None:
            scores.append({'event': ev, 'n_epochs': 0, 'errp_score': 0.0})
            ax.set_title(f"'{ev}': no usable epochs")
            continue
        times, mean_uV, n = result
        sc = _ne_pe_score(times, mean_uV)
        scores.append({'event': ev, 'n_epochs': n, 'errp_score': round(sc, 3)})
        ax.plot(times * 1000, mean_uV, color='steelblue')
        ax.axhline(0, color='k', lw=0.5)
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.axvspan(200, 350, color='red', alpha=0.08)
        ax.axvspan(300, 500, color='green', alpha=0.08)
        ax.set_title(f"'{ev}'  n={n}  score={sc:.2f}uV")
        ax.set_ylabel(f'{channel} (uV)')
    axes[-1, 0].set_xlabel('Time relative to event (ms)')
    fig.suptitle('Grand-average ERP per candidate '
                 '(red=Ne window, green=Pe window)', y=1.01)
    fig.tight_layout()
    fig.savefig(f'{save_diag_path}/candidate_erps.png', dpi=120, bbox_inches='tight')
    plt.show()

    scores_df = pd.DataFrame(scores).sort_values('errp_score', ascending=False)
    scores_df.to_csv(f'{save_diag_path}/erp_scores.csv', index=False)
    print('=== ERP signature scores (higher = stronger Ne-Pe) ===')
    print(scores_df.to_string(index=False))

    top = scores_df.iloc[0]
    if top['errp_score'] < 1.0:
        print('\n!! No candidate shows a clear Ne-Pe complex (score < 1 uV).')
        print('   Fallback: rarer plausible events labeled 1 (error), '
              'commoner labeled 0 (non-error).')
        per_med = {r['event']: r['median_count']
                   for _, r in summary_df.iterrows() if r['plausible_feedback']}
        med_of_meds = np.median(list(per_med.values()))
        label_map = {ev: (1 if cnt < med_of_meds else 0)
                     for ev, cnt in per_med.items()}
    else:
        error_ev = top['event']
        others = scores_df[scores_df['event'] != error_ev]
        if len(others) > 0:
            noerr_ev = others.iloc[0]['event']
            label_map = {noerr_ev: 0, error_ev: 1}
        else:
            label_map = {error_ev: 1}

    print(f'\n>>> AUTO-DETECTED Config.ERRP_EVENT_LABEL_MAP = {label_map}')
    print('    Override by setting MANUAL_ERRP_EVENT_LABEL_MAP before this cell.')
    return label_map

_SESSION_ORDER: Dict[str, int] = {
    'calib': 1, 'corl1': 2, 'corl2': 3, 'corl3': 4, 'corl4': 5}

def index_coadaptation_dataset(data_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """Return {subject_id: [(session_num, filepath), ...]} sorted by session_num."""
    idx: Dict = defaultdict(list)
    for subj_dir in sorted(glob.glob(os.path.join(data_dir, "s*"))):
        if not os.path.isdir(subj_dir):
            continue
        sname = os.path.basename(subj_dir)
        sid   = sname.upper()
        for sess_name, sess_num in _SESSION_ORDER.items():
            set_file = os.path.join(subj_dir, f"{sname}_{sess_name}.set")
            if os.path.isfile(set_file):
                idx[sid].append((sess_num, set_file))
    for k in idx:
        idx[k].sort()
    return dict(idx)

_MIN_TRIALS_PER_SESSION = 30

_MAX_TRIALS_PER_SESSION = 600

_MIN_MINORITY_FRACTION  = 0.03

def extract_labels_from_set(
        set_filepath: str,
        event_label_map: Dict[str, int],
        inspect_events: bool = False) -> Optional[List[Tuple[int, int]]]:
    """Extract (trial_idx, label) pairs; returns None on sanity-gate failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            raw = mne.io.read_raw_eeglab(set_filepath, preload=False, verbose=False)
    except Exception as exc:
        print(f"    [skip] {os.path.basename(set_filepath)}: {exc}")
        return None

    if inspect_events:
        unique = Counter(a['description'] for a in raw.annotations)
        print(f"    Event counts in {os.path.basename(set_filepath)}: {dict(unique)}")

    labels: List[Tuple[int, int]] = []
    i = 0
    for annot in raw.annotations:
        if annot['description'] in event_label_map:
            labels.append((i, event_label_map[annot['description']]))
            i += 1

    if not labels:
        return None

    n = len(labels)
    if not (_MIN_TRIALS_PER_SESSION <= n <= _MAX_TRIALS_PER_SESSION):
        print(f"    [skip] {os.path.basename(set_filepath)}: {n} trials outside "
              f"[{_MIN_TRIALS_PER_SESSION}, {_MAX_TRIALS_PER_SESSION}] "
              f"— wrong event type for this session.")
        return None

    cls_counts = Counter(lbl for _, lbl in labels)
    if len(cls_counts) < 2:
        print(f"    [skip] {os.path.basename(set_filepath)}: single class "
              f"{dict(cls_counts)} — cannot train.")
        return None

    minority_frac = min(cls_counts.values()) / n
    if minority_frac < _MIN_MINORITY_FRACTION:
        print(f"    [warn] {os.path.basename(set_filepath)}: minority class "
              f"{minority_frac:.2%} very low — keeping but flag.")

    return labels

def build_coadaptation_mapping(
        subject_index: Dict[str, List[Tuple[int, str]]],
        event_label_map: Dict[str, int]) -> Dict[str, List[Dict]]:
    """Build dataset_mapping; degenerate sessions are dropped with a printed reason."""
    mapping: Dict[str, List[Dict]] = {}
    print(f"Building dataset mapping for {len(subject_index)} subjects...")
    for i, (sid, sessions) in enumerate(sorted(subject_index.items())):
        subj_sessions: List[Dict] = []
        for sess_num, fp in sessions:
            inspect = (i == 0 and sess_num == sessions[0][0])
            labels = extract_labels_from_set(fp, event_label_map,
                                             inspect_events=inspect)
            if labels is None:
                continue
            subj_sessions.append({'session_num': sess_num, 'filepath': fp,
                                  'labels': labels})
        if subj_sessions:
            mapping[sid] = subj_sessions
        else:
            print(f"  [drop subject] {sid}: no usable sessions")
    return mapping

def load_continuous_eeg(filepath: str) -> Tuple[pd.DataFrame, float]:
    """Load EEGLAB .set EEG file. Returns (DataFrame, sfreq).

    DataFrame columns: EEG channel columns + 'Time' + 'FeedBackEvent'.
    FeedBackEvent = 1 at each feedback event onset sample (0 elsewhere),
    using Config.ERRP_EVENT_LABEL_MAP to identify feedback events.

    Data is converted from MNE internal Volts to microvolts (×1e6) so that
    the downstream artifact rejection threshold (100 µV) is applied correctly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    data, times = raw.get_data(return_times=True)  # (n_ch, n_samples), Volts
    data = data * 1e6                               # V → µV (pipeline expects µV)

    df = pd.DataFrame(data.T, columns=raw.ch_names)   # (n_samples, n_ch)
    df['Time'] = times

    n_samp    = len(times)
    fb_col    = np.zeros(n_samp, dtype=int)
    event_map = getattr(Config, 'ERRP_EVENT_LABEL_MAP', {})
    first_samp = int(raw.first_samp)
    for annot in raw.annotations:
        if annot['description'] in event_map:
            idx = int(round(annot['onset'] * sfreq)) - first_samp
            if 0 <= idx < n_samp:
                fb_col[idx] = 1
    df['FeedBackEvent'] = fb_col
    return df, sfreq

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
