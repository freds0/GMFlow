"""Pre-process OpenBHB volumes: downsample to target resolution and save as .pt.

Usage:
    python tools/prepare_openbhb.py \
        --data_root data/openbhb/train/quasiraw_3d \
        --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
        --output_dir data/openbhb/train_cache_64 \
        --target_shape 64 64 64 \
        --clip_range 3.0
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def normalize_volume(vol, clip_range=3.0):
    """Z-score normalize, clip, and rescale to [-1, 1]."""
    mean = vol.mean()
    std = vol.std().clamp(min=1e-6)
    vol = (vol - mean) / std
    vol = vol.clamp(-clip_range, clip_range)
    vol = vol / clip_range
    return vol


def main():
    parser = argparse.ArgumentParser(description='Prepare OpenBHB dataset')
    parser.add_argument('--data_root', type=str,
                        default='data/openbhb/train/quasiraw_3d')
    parser.add_argument('--metadata', type=str,
                        default='data/openbhb/train/quasiraw_3d/metadata.tsv',
                        help='Path to metadata.tsv or train.tsv')
    parser.add_argument('--output_dir', type=str,
                        default='data/openbhb/train_cache_64')
    parser.add_argument('--target_shape', type=int, nargs=3,
                        default=[64, 64, 64])
    parser.add_argument('--npy_suffix', type=str, default='_quasiraw_3d',
                        help='Suffix in .npy filenames (e.g. _quasiraw_3d)')
    parser.add_argument('--clip_range', type=float, default=3.0)
    parser.add_argument('--split', type=str, default=None,
                        help='Filter by split column (e.g. "train"). '
                             'Only needed if metadata has a split column.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = pd.read_csv(args.metadata, sep='\t')
    if args.split and 'split' in metadata.columns:
        metadata = metadata[metadata['split'] == args.split]
        print(f'Filtered to split={args.split}: {len(metadata)} subjects')
    target_shape = tuple(args.target_shape)

    processed = 0
    skipped = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc='Processing'):
        participant_id = str(row['participant_id'])
        out_path = os.path.join(args.output_dir, f'{participant_id}.pt')

        if os.path.exists(out_path):
            skipped += 1
            continue

        npy_path = os.path.join(
            args.data_root, f'{participant_id}{args.npy_suffix}.npy')
        if not os.path.exists(npy_path):
            print(f'Warning: {npy_path} not found, skipping.')
            continue

        vol = np.load(npy_path).astype(np.float32)
        vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        vol = F.interpolate(vol, size=target_shape, mode='trilinear', align_corners=False)
        vol = vol.squeeze(0)  # (1, D, H, W)
        vol = normalize_volume(vol, clip_range=args.clip_range)

        torch.save(vol, out_path)
        processed += 1

    print(f'Done. Processed: {processed}, Skipped (already cached): {skipped}')


if __name__ == '__main__':
    main()