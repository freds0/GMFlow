import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mmcv

from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC
from mmgen.datasets.builder import DATASETS
from mmgen.utils import get_root_logger


@DATASETS.register_module()
class OpenBHB(Dataset):
    """OpenBHB Brain MRI dataset for 3D generation conditioned on age.

    Loads preprocessed .pt volumes (or raw .npy + downsample on the fly),
    normalizes to [-1, 1], and provides age as a continuous condition.

    Args:
        data_root (str): Root directory of the dataset.
        metadata_path (str): Path to metadata.tsv with age info.
        target_shape (tuple): Target volume shape after downsampling.
        random_flip (bool): Random left-right flip (sagital axis).
        age_min (float): Minimum age for normalization.
        age_max (float): Maximum age for normalization.
        negative_age (float): Sentinel value for unconditional (CFG null).
        use_cache (bool): If True, load from preprocessed .pt files.
        cache_dir (str): Directory with preprocessed .pt cache files.
        test_mode (bool): If True, generate dummy data for validation.
        num_test_volumes (int): Number of test volumes to generate.
        clip_range (float): Clip range for z-score normalization.
    """

    def __init__(
            self,
            data_root='data/openbhb/train/quasiraw_3d',
            metadata_path='data/openbhb/train/quasiraw_3d/metadata.tsv',
            npy_suffix='_quasiraw_3d',
            target_shape=(64, 64, 64),
            random_flip=True,
            age_min=6.0,
            age_max=86.0,
            negative_age=-1.0,
            use_cache=True,
            cache_dir='data/openbhb/train_cache_64',
            test_mode=False,
            num_test_volumes=100,
            clip_range=3.0):
        super().__init__()
        self.data_root = data_root
        self.metadata_path = metadata_path
        self.target_shape = tuple(target_shape)
        self.random_flip = random_flip
        self.age_min = age_min
        self.age_max = age_max
        self.age_range = age_max - age_min
        self.negative_age = negative_age
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.test_mode = test_mode
        self.num_test_volumes = num_test_volumes
        self.npy_suffix = npy_suffix
        self.clip_range = clip_range

        if not test_mode:
            metadata = pd.read_csv(metadata_path, sep='\t')
            self.subjects = []
            for _, row in metadata.iterrows():
                participant_id = str(row['participant_id'])
                age = float(row['age'])
                cache_path = os.path.join(cache_dir, f'{participant_id}.pt')
                raw_path = os.path.join(
                    data_root, f'{participant_id}{npy_suffix}.npy')
                if self.use_cache and os.path.exists(cache_path):
                    vol_path = cache_path
                elif os.path.exists(raw_path):
                    vol_path = raw_path
                else:
                    vol_path = None

                if vol_path is not None:
                    self.subjects.append(dict(
                        participant_id=participant_id,
                        age=age,
                        path=vol_path,
                        cache_path=cache_path,
                        raw_path=raw_path))

            logger = get_root_logger()
            mmcv.print_log(f'OpenBHB data root: {self.data_root}', logger=logger)
            mmcv.print_log(f'OpenBHB cache dir: {self.cache_dir}', logger=logger)
            mmcv.print_log(f'Number of subjects: {len(self.subjects)}', logger=logger)

    def __len__(self):
        return self.num_test_volumes if self.test_mode else len(self.subjects)

    def _normalize_volume(self, vol):
        """Z-score normalize, clip, and rescale to [-1, 1]."""
        mean = vol.mean()
        std = vol.std().clamp(min=1e-6)
        vol = (vol - mean) / std
        vol = vol.clamp(-self.clip_range, self.clip_range)
        vol = vol / self.clip_range  # now in [-1, 1]
        return vol

    def _load_and_preprocess(self, path):
        """Load a raw .npy volume and downsample to target_shape.

        Normalization is applied centrally in ``__getitem__`` so that the
        cached and on-the-fly paths share the exact same intensity statistics.
        """
        vol = np.load(path).astype(np.float32)
        vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        vol = F.interpolate(vol, size=self.target_shape, mode='trilinear', align_corners=False)
        vol = vol.squeeze(0)  # (1, D, H, W)
        return vol

    def _load_cached(self, path):
        """Load a preprocessed .pt volume (normalized centrally in __getitem__)."""
        return torch.load(path, weights_only=True)

    def _normalize_age(self, age):
        """Normalize age to [0, 1]."""
        return (age - self.age_min) / self.age_range

    def __getitem__(self, idx):
        data = dict(ids=DC(idx, cpu_only=True))

        if self.test_mode:
            generator = torch.Generator().manual_seed(idx)
            age_norm = torch.rand((), generator=generator).item()
            age = age_norm * self.age_range + self.age_min
            data.update(
                age=torch.tensor(age_norm, dtype=torch.float32),
                name=DC(f'test_age{age:.1f}', cpu_only=True),
                negative_age=torch.tensor(self.negative_age, dtype=torch.float32))
            return data

        subject = self.subjects[idx]

        if self.use_cache and subject['path'].endswith('.pt'):
            vol = self._load_cached(subject['path'])
        else:
            vol = self._load_and_preprocess(subject['path'])

        # Normalize centrally so cached (.pt) and on-the-fly (.npy) paths are
        # identically distributed. Z-score is invariant to affine rescaling, so
        # a cache stored in [0, 1] and a raw volume yield the same result here.
        vol = self._normalize_volume(vol)

        # Random left-right flip (sagital axis, last dim = W)
        if self.random_flip and np.random.rand() < 0.5:
            vol = vol.flip(-1)

        age_norm = self._normalize_age(subject['age'])

        data.update(
            volumes=vol,  # (1, D, H, W)
            age=torch.tensor(age_norm, dtype=torch.float32),
            name=DC(subject['participant_id'], cpu_only=True),
            negative_age=torch.tensor(self.negative_age, dtype=torch.float32))

        return data
