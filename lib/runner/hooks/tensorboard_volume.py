import os

import numpy as np
import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class VolumeTensorboardHook(Hook):
    """Log generated 3D volume slices to TensorBoard.

    The generative eval hook writes generated volumes as .npy files. This hook
    watches that directory after validation and logs axial/coronal/sagittal
    center slices as image grids.
    """

    def __init__(
            self,
            image_dir,
            tag='samples/volumes',
            max_images=4,
            interval=100,
            boundary_period=4):
        self.image_dir = image_dir
        self.tag = tag
        self.max_images = max_images
        self.interval = interval
        self.boundary_period = boundary_period
        if self.boundary_period is not None and self.boundary_period < 2:
            raise ValueError('boundary_period must be at least 2')
        self.writer = None
        self._last_logged_iter = -1

    @staticmethod
    def _to_volume(array):
        array = np.asarray(array)
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3:
            return None
        return array.astype(np.float32)

    @staticmethod
    def _normalize_slice(image, low, high):
        image = np.nan_to_num(image.astype(np.float32), nan=low, posinf=high, neginf=low)
        if high <= low:
            return np.zeros_like(image, dtype=np.float32)
        image = np.clip((image - low) / (high - low), 0.0, 1.0)
        return image.astype(np.float32)

    @staticmethod
    def _periodic_boundary_ratio(volume, period):
        ratios = []
        for axis in range(3):
            gradient = np.abs(np.diff(volume, axis=axis))
            indices = np.arange(gradient.shape[axis])
            boundary = (indices + 1) % period == 0
            if not boundary.any() or boundary.all():
                continue
            boundary_mean = np.take(
                gradient, indices[boundary], axis=axis).mean()
            interior_mean = np.take(
                gradient, indices[~boundary], axis=axis).mean()
            ratios.append(
                float(boundary_mean / max(interior_mean, 1e-12)))
        return float(np.mean(ratios)) if ratios else 1.0

    @staticmethod
    def _invalid_slice(height, width):
        image = np.zeros((height, width), dtype=np.float32)
        image[::4, :] = 1.0
        image[:, ::4] = 1.0
        return image

    def before_run(self, runner):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(os.path.join(runner.work_dir, 'tf_logs'))

    def after_train_iter(self, runner):
        if self.writer is None or self._last_logged_iter == runner.iter:
            return
        if self.interval > 0 and not self.every_n_iters(runner, self.interval):
            return
        if not os.path.isdir(self.image_dir):
            return

        npy_files = sorted(
            os.path.join(self.image_dir, name)
            for name in os.listdir(self.image_dir)
            if name.endswith('.npy'))
        if not npy_files:
            return

        images = []
        nonfinite_fracs = []
        finite_means = []
        finite_stds = []
        boundary_ratios = []
        for path in npy_files[:self.max_images]:
            volume = self._to_volume(np.load(path))
            if volume is None:
                continue

            finite = np.isfinite(volume)
            nonfinite_fracs.append(1.0 - float(finite.mean()))
            if finite.any():
                finite_values = volume[finite]
                low, high = np.percentile(finite_values, [1.0, 99.0])
                finite_means.append(float(finite_values.mean()))
                finite_stds.append(float(finite_values.std()))
                if self.boundary_period is not None:
                    boundary_ratios.append(
                        self._periodic_boundary_ratio(
                            np.nan_to_num(volume), self.boundary_period))
            else:
                low, high = 0.0, 1.0
                finite_means.append(float('nan'))
                finite_stds.append(float('nan'))

            d, h, w = volume.shape
            if finite.any():
                slices = [
                    volume[d // 2, :, :],
                    volume[:, h // 2, :],
                    volume[:, :, w // 2],
                ]
                slices = [self._normalize_slice(image, low, high) for image in slices]
            else:
                slices = [
                    self._invalid_slice(h, w),
                    self._invalid_slice(d, w),
                    self._invalid_slice(d, h),
                ]
            for image in slices:
                images.append(torch.from_numpy(image).unsqueeze(0))

        if nonfinite_fracs:
            self.writer.add_scalar(
                self.tag + '/nonfinite_fraction',
                float(np.mean(nonfinite_fracs)),
                runner.iter)
        if boundary_ratios:
            self.writer.add_scalar(
                self.tag + '/patch_boundary_ratio',
                float(np.mean(boundary_ratios)),
                runner.iter)
        if finite_means and np.isfinite(finite_means).any():
            self.writer.add_scalar(
                self.tag + '/finite_mean',
                float(np.nanmean(finite_means)),
                runner.iter)
            self.writer.add_scalar(
                self.tag + '/finite_std',
                float(np.nanmean(finite_stds)),
                runner.iter)

        if images:
            grid = torch.stack(images, dim=0)
            self.writer.add_images(self.tag, grid, runner.iter)
            self.writer.flush()
            self._last_logged_iter = runner.iter

    def after_run(self, runner):
        if self.writer is not None:
            self.writer.close()
