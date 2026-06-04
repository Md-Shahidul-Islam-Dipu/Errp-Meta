"""Temporal feature extraction + LOSO-isolated PCA.

Auto-ported verbatim from the legacy notebooks by _build_pkg.py.
"""
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..progress import tqdm

from ..config import Config

def extract_temporal_features(epochs: np.ndarray) -> np.ndarray:
    """Flatten (N, C, T) → (N, C*T). Temporal waveform features for meta-learning."""
    n, c, t = epochs.shape
    return epochs.reshape(n, c * t)

def extract_features_all_subjects(ppd: Dict[str, Dict]) -> Dict[str, Dict]:
    """Extract and return temporal features for all subjects."""
    out = {}
    for sid, data in tqdm(sorted(ppd.items()), desc="Feature extraction"):
        out[sid] = {'subject_id': sid,
                    'features': extract_temporal_features(data['epochs']).astype(np.float32),
                    'labels':   data['labels']}
    return out

class PCAReducer:
    """Fit StandardScaler + whitened PCA on training data; transform any data."""
    def __init__(self, n_components: int = 32, seed: int = 42):
        self.scaler    = StandardScaler()
        self.pca       = PCA(n_components=n_components, whiten=True, random_state=seed)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'PCAReducer':
        self.pca.fit(self.scaler.fit_transform(X))
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X)).astype(np.float32)

def apply_pca_loso(subjects_features: Dict[str, Dict],
                   n_components: int = Config.PCA_COMPONENTS) -> Dict[str, Dict]:
    """Build LOSO PCA splits. Each fold's PCA fitted on N-1 training subjects.

    Each fold contains:
      test            : {features (PCA-reduced), labels}
      pca             : fitted PCAReducer for this fold
      train_subjects  : list of training subject IDs
      subjects_features: reference to raw feature dict (for training transforms)
    """
    sids = sorted(subjects_features.keys())
    splits = {}
    for test_sid in tqdm(sids, desc="LOSO PCA"):
        train_sids = [s for s in sids if s != test_sid]
        train_X    = np.concatenate([subjects_features[s]['features'] for s in train_sids])
        reducer    = PCAReducer(n_components=n_components).fit(train_X)
        splits[test_sid] = {
            'test'             : {'features': reducer.transform(subjects_features[test_sid]['features']),
                                  'labels'  : subjects_features[test_sid]['labels']},
            'pca'              : reducer,
            'train_subjects'   : train_sids,
            'subjects_features': subjects_features,
        }
    return splits
