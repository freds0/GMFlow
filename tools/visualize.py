"""Visualize generated 3D brain MRI volumes.

Usage:
    # Visualize all .npy files in a directory (grid of center slices):
    python tools/visualize.py output/samples/

    # Visualize specific files:
    python tools/visualize.py output/samples/age010.0_seed42.npy output/samples/age075.0_seed42.npy

    # Compare real vs generated:
    python tools/visualize.py \
        --real /home/fred/Projetos/Einstein/openbhb_train_sample/train_cache_64/100053248969.pt \
        --generated output/samples/age025.0_seed42.npy

    # Interactive 3D slice explorer:
    python tools/visualize.py output/samples/age050.0_seed42.npy --interactive

    # Save figure without showing:
    python tools/visualize.py output/samples/ --save viz.png --no_show
"""

import os
import argparse
import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch


def load_volume(path):
    """Load a volume from .npy or .pt file. Returns numpy array (D, H, W)."""
    if path.endswith('.pt'):
        vol = torch.load(path, map_location='cpu', weights_only=True)
        vol = vol.numpy()
    elif path.endswith('.npy'):
        vol = np.load(path)
    else:
        raise ValueError(f'Unsupported file format: {path}')

    # Remove channel dim if present: (1, D, H, W) -> (D, H, W)
    while vol.ndim > 3:
        vol = vol[0]

    return vol.astype(np.float32)


def get_center_slices(vol):
    """Get 3 orthogonal center slices from a (D, H, W) volume."""
    d, h, w = vol.shape
    return {
        'Axial': vol[d // 2, :, :],
        'Coronal': vol[:, h // 2, :],
        'Sagittal': vol[:, :, w // 2],
    }


def normalize_for_display(img, vmin=None, vmax=None):
    """Normalize image to [0, 1] for display."""
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def extract_age_from_filename(path):
    """Try to extract age from filename like 'age025.0_seed42.npy'."""
    name = os.path.basename(path)
    if name.startswith('age'):
        try:
            age_str = name.split('_')[0].replace('age', '')
            return float(age_str)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Visualization modes
# ---------------------------------------------------------------------------

def viz_grid(volumes, labels, args):
    """Show a grid: rows=volumes, columns=axial/coronal/sagittal."""
    n = len(volumes)
    views = ['Axial', 'Coronal', 'Sagittal']

    # Compute global range for consistent colormap
    if args.shared_colormap:
        all_vals = np.concatenate([v.ravel() for v in volumes])
        vmin, vmax = np.percentile(all_vals, [1, 99])
    else:
        vmin, vmax = None, None

    fig, axes = plt.subplots(n, 3, figsize=(3 * 3.5, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (vol, label) in enumerate(zip(volumes, labels)):
        slices = get_center_slices(vol)
        for j, view in enumerate(views):
            img = slices[view]
            if not args.shared_colormap:
                vmin_i, vmax_i = np.percentile(img, [1, 99])
            else:
                vmin_i, vmax_i = vmin, vmax

            ax = axes[i, j]
            ax.imshow(img.T if view != 'Axial' else img,
                      cmap=args.cmap, vmin=vmin_i, vmax=vmax_i,
                      origin='lower', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(view, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(label, fontsize=12)

    plt.suptitle('Generated Brain MRI Volumes (center slices)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved figure to {args.save}')

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def viz_compare(real_paths, gen_paths, args):
    """Side-by-side comparison of real vs generated volumes."""
    n = max(len(real_paths), len(gen_paths))
    views = ['Axial', 'Coronal', 'Sagittal']

    fig, axes = plt.subplots(n, 6, figsize=(6 * 3, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        # Real
        if i < len(real_paths):
            vol_r = load_volume(real_paths[i])
            slices_r = get_center_slices(vol_r)
            label_r = os.path.basename(real_paths[i]).split('.')[0]
        else:
            slices_r = None

        # Generated
        if i < len(gen_paths):
            vol_g = load_volume(gen_paths[i])
            slices_g = get_center_slices(vol_g)
            label_g = os.path.basename(gen_paths[i]).split('.')[0]
        else:
            slices_g = None

        for j, view in enumerate(views):
            # Real column
            ax_r = axes[i, j]
            if slices_r is not None:
                img = slices_r[view]
                v1, v2 = np.percentile(img, [1, 99])
                ax_r.imshow(img.T if view != 'Axial' else img,
                            cmap=args.cmap, vmin=v1, vmax=v2,
                            origin='lower', aspect='equal')
                if j == 0:
                    ax_r.set_ylabel(f'Real\n{label_r}', fontsize=10)
            ax_r.set_xticks([])
            ax_r.set_yticks([])
            if i == 0:
                ax_r.set_title(f'Real {view}', fontsize=11, fontweight='bold')

            # Generated column
            ax_g = axes[i, j + 3]
            if slices_g is not None:
                img = slices_g[view]
                v1, v2 = np.percentile(img, [1, 99])
                ax_g.imshow(img.T if view != 'Axial' else img,
                            cmap=args.cmap, vmin=v1, vmax=v2,
                            origin='lower', aspect='equal')
                if j == 0:
                    ax_g.set_ylabel(f'Gen\n{label_g}', fontsize=10)
            ax_g.set_xticks([])
            ax_g.set_yticks([])
            if i == 0:
                ax_g.set_title(f'Gen {view}', fontsize=11, fontweight='bold')

    plt.suptitle('Real vs Generated', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved figure to {args.save}')

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def viz_interactive(vol, label, args):
    """Interactive 3D slice explorer with sliders for each axis."""
    d, h, w = vol.shape
    vmin, vmax = np.percentile(vol, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)

    # Initial slices at center
    im_ax = axes[0].imshow(vol[d // 2], cmap=args.cmap, vmin=vmin, vmax=vmax,
                            origin='lower', aspect='equal')
    im_cor = axes[1].imshow(vol[:, h // 2, :].T, cmap=args.cmap, vmin=vmin, vmax=vmax,
                             origin='lower', aspect='equal')
    im_sag = axes[2].imshow(vol[:, :, w // 2].T, cmap=args.cmap, vmin=vmin, vmax=vmax,
                             origin='lower', aspect='equal')

    axes[0].set_title('Axial')
    axes[1].set_title('Coronal')
    axes[2].set_title('Sagittal')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Sliders
    ax_sl_d = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_sl_h = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_sl_w = plt.axes([0.15, 0.05, 0.7, 0.03])

    sl_d = Slider(ax_sl_d, 'Axial (D)', 0, d - 1, valinit=d // 2, valstep=1)
    sl_h = Slider(ax_sl_h, 'Coronal (H)', 0, h - 1, valinit=h // 2, valstep=1)
    sl_w = Slider(ax_sl_w, 'Sagittal (W)', 0, w - 1, valinit=w // 2, valstep=1)

    def update(_):
        im_ax.set_data(vol[int(sl_d.val)])
        im_cor.set_data(vol[:, int(sl_h.val), :].T)
        im_sag.set_data(vol[:, :, int(sl_w.val)].T)
        fig.canvas.draw_idle()

    sl_d.on_changed(update)
    sl_h.on_changed(update)
    sl_w.on_changed(update)

    fig.suptitle(f'{label}  |  shape={vol.shape}  range=[{vol.min():.3f}, {vol.max():.3f}]',
                 fontsize=13)
    plt.show()


def viz_montage(vol, label, args):
    """Show montage of evenly-spaced axial slices."""
    d, h, w = vol.shape
    n_slices = min(args.montage_slices, d)
    indices = np.linspace(0, d - 1, n_slices, dtype=int)

    cols = min(8, n_slices)
    rows = (n_slices + cols - 1) // cols

    vmin, vmax = np.percentile(vol, [1, 99])

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        r, c = i // cols, i % cols
        axes[r, c].imshow(vol[idx], cmap=args.cmap, vmin=vmin, vmax=vmax,
                          origin='lower', aspect='equal')
        axes[r, c].set_title(f'slice {idx}', fontsize=9)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    # Hide empty axes
    for i in range(n_slices, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')

    fig.suptitle(f'{label} - Axial Montage', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved figure to {args.save}')

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def viz_histogram(volumes, labels, args):
    """Show voxel intensity histograms for all volumes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for vol, label in zip(volumes, labels):
        vals = vol.ravel()
        ax.hist(vals, bins=100, alpha=0.5, label=label, density=True)

    ax.set_xlabel('Voxel Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Voxel Intensity Distribution')
    ax.legend()

    if args.save:
        save_path = args.save.replace('.png', '_hist.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved histogram to {save_path}')

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_files(inputs):
    """Resolve inputs to a list of .npy/.pt file paths."""
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, '*.npy'))))
            files.extend(sorted(glob.glob(os.path.join(inp, '*.pt'))))
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            print(f'Warning: {inp} not found, skipping')
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Visualize generated 3D brain MRI volumes',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('inputs', nargs='*', default=[],
                        help='Paths to .npy/.pt files or directories')

    # Comparison mode
    parser.add_argument('--real', nargs='+', default=None,
                        help='Real volume(s) for side-by-side comparison')
    parser.add_argument('--generated', nargs='+', default=None,
                        help='Generated volume(s) for comparison')

    # Visualization mode
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive slice explorer (single volume)')
    parser.add_argument('--montage', action='store_true',
                        help='Show axial slice montage')
    parser.add_argument('--montage_slices', type=int, default=16,
                        help='Number of slices in montage')
    parser.add_argument('--histogram', action='store_true',
                        help='Show voxel intensity histograms')

    # Display options
    parser.add_argument('--cmap', type=str, default='gray',
                        help='Colormap (default: gray)')
    parser.add_argument('--shared_colormap', action='store_true',
                        help='Use shared intensity range across all volumes')

    # Output
    parser.add_argument('--save', type=str, default=None,
                        help='Save figure to file (e.g., viz.png)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display (only save)')

    args = parser.parse_args()

    # Comparison mode
    if args.real is not None and args.generated is not None:
        print(f'Comparison mode: {len(args.real)} real vs {len(args.generated)} generated')
        viz_compare(args.real, args.generated, args)
        return

    # Collect files from inputs
    files = collect_files(args.inputs)
    if not files:
        parser.print_help()
        print('\nError: no .npy or .pt files found.')
        return

    print(f'Loading {len(files)} volume(s)...')
    volumes = []
    labels = []
    for f in files:
        vol = load_volume(f)
        volumes.append(vol)
        age = extract_age_from_filename(f)
        if age is not None:
            labels.append(f'Age {age:.0f}yr')
        else:
            labels.append(os.path.basename(f).split('.')[0])
        print(f'  {os.path.basename(f)}: shape={vol.shape} '
              f'range=[{vol.min():.3f}, {vol.max():.3f}]')

    # Interactive mode (single volume)
    if args.interactive:
        if len(volumes) > 1:
            print(f'Interactive mode: showing first volume ({labels[0]})')
        viz_interactive(volumes[0], labels[0], args)
        return

    # Montage mode (single volume)
    if args.montage:
        if len(volumes) > 1:
            print(f'Montage mode: showing first volume ({labels[0]})')
        viz_montage(volumes[0], labels[0], args)
        return

    # Histogram
    if args.histogram:
        viz_histogram(volumes, labels, args)
        return

    # Default: grid of center slices
    viz_grid(volumes, labels, args)


if __name__ == '__main__':
    main()
