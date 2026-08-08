#!/usr/bin/env python3
"""Summarize GMFlow3D architecture-ablation training runs.

The report deliberately keeps optimization and image-proxy metrics separate:
losses with different objectives are not directly comparable, and none of the
proxy metrics replaces clinical or anatomical validation.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


SAMPLE_PATTERN = re.compile(
    r"iter(?P<iteration>\d+)_age(?P<age>-?\d+(?:\.\d+)?)\.npy$")
LOAD_PATTERN = re.compile(
    r"Model: loaded (?P<loaded>\d+) tensors, "
    r"skipped (?P<skipped>\d+), missing (?P<missing>\d+)")
PARAM_PATTERN = re.compile(r"Model parameters: (?P<parameters>[\d,]+)")


def parse_run_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--run must use NAME=PATH, for example full=work_dirs/run/full")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run requires non-empty NAME and PATH")
    return name, Path(path)


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json_log(path):
    entries = []
    if not path.is_file():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def read_checkpoint_compatibility(path):
    result = dict(loaded=None, skipped=None, missing=None, parameters=None)
    if not path.is_file():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    match = LOAD_PATTERN.search(text)
    if match:
        result.update(
            loaded=int(match.group("loaded")),
            skipped=int(match.group("skipped")),
            missing=int(match.group("missing")),
        )
    match = PARAM_PATTERN.search(text)
    if match:
        result["parameters"] = int(match.group("parameters").replace(",", ""))
    return result


def read_tensorboard_scalars(run_dir):
    values = {}
    event_files = sorted((run_dir / "tb").glob("events.*"))
    if not event_files:
        return values
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        accumulator = EventAccumulator(
            str(event_files[-1]), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = set(accumulator.Tags().get("scalars", []))
        for tag in (
                "train/loss_auxiliary",
                "train/loss_gm_mean",
                "train/loss_voxel_gradient"):
            if tag in tags:
                events = accumulator.Scalars(tag)
                if events:
                    values[tag] = finite_float(events[-1].value)
    except Exception as error:
        values["tensorboard_error"] = str(error)
    return values


def to_volume(array):
    volume = np.asarray(array, dtype=np.float32).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D sample, got shape {volume.shape}")
    return volume


def periodic_boundary_ratio(volume, period):
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
        ratios.append(float(boundary_mean / max(interior_mean, 1e-12)))
    return float(np.mean(ratios)) if ratios else None


def haar_detail_ratio(volume):
    if any(size % 2 for size in volume.shape):
        return None
    bands = [volume]
    scale = math.sqrt(2.0)
    for axis in range(3):
        transformed = []
        for band in bands:
            even_index = [slice(None)] * 3
            odd_index = [slice(None)] * 3
            even_index[axis] = slice(0, None, 2)
            odd_index[axis] = slice(1, None, 2)
            even = band[tuple(even_index)]
            odd = band[tuple(odd_index)]
            transformed.extend([(even + odd) / scale, (even - odd) / scale])
        bands = transformed

    rms = np.asarray(
        [math.sqrt(float(np.mean(np.square(band)))) for band in bands])
    return float(rms[1:].mean() / max(rms[0], 1e-12))


def sample_metrics(run_dir, boundary_period, detail_reference):
    sample_paths = []
    for path in (run_dir / "samples").glob("*.npy"):
        match = SAMPLE_PATTERN.match(path.name)
        if match:
            sample_paths.append(
                (int(match.group("iteration")), float(match.group("age")), path))

    result = dict(
        sample_iteration=None,
        num_samples=0,
        nonfinite_fraction=None,
        boundary_ratio=None,
        detail_to_lll=None,
        detail_abs_error=None,
        age_l1_delta=None,
        intensity_std=None,
    )
    if not sample_paths:
        return result

    latest_iteration = max(item[0] for item in sample_paths)
    latest = sorted(
        (item for item in sample_paths if item[0] == latest_iteration),
        key=lambda item: item[1])

    volumes = []
    nonfinite = []
    boundary = []
    detail = []
    intensity_std = []
    for _, age, path in latest:
        try:
            volume = to_volume(np.load(path))
        except (OSError, ValueError):
            continue
        finite = np.isfinite(volume)
        nonfinite.append(1.0 - float(finite.mean()))
        clean = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=-1.0)
        boundary_value = periodic_boundary_ratio(clean, boundary_period)
        detail_value = haar_detail_ratio(clean)
        if boundary_value is not None:
            boundary.append(boundary_value)
        if detail_value is not None:
            detail.append(detail_value)
        intensity_std.append(float(clean.std()))
        volumes.append((age, clean))

    age_deltas = []
    for (_, first), (_, second) in zip(volumes, volumes[1:]):
        if first.shape == second.shape:
            age_deltas.append(float(np.mean(np.abs(second - first))))

    detail_mean = float(np.mean(detail)) if detail else None
    result.update(
        sample_iteration=latest_iteration,
        num_samples=len(volumes),
        nonfinite_fraction=float(np.mean(nonfinite)) if nonfinite else None,
        boundary_ratio=float(np.mean(boundary)) if boundary else None,
        detail_to_lll=detail_mean,
        detail_abs_error=(
            abs(detail_mean - detail_reference)
            if detail_mean is not None else None),
        age_l1_delta=float(np.mean(age_deltas)) if age_deltas else None,
        intensity_std=(
            float(np.mean(intensity_std)) if intensity_std else None),
    )
    return result


def status_for_run(run_dir):
    status_path = run_dir / "exit_status.txt"
    if status_path.is_file():
        value = status_path.read_text(encoding="utf-8").strip()
        return "complete" if value == "0" else f"failed({value})"
    if (run_dir / "train_log.jsonl").is_file():
        return "unknown"
    return "missing"


def summarize_run(name, run_dir, boundary_period, detail_reference):
    entries = read_json_log(run_dir / "train_log.jsonl")
    compatibility = read_checkpoint_compatibility(run_dir / "launcher.log")
    tensorboard = read_tensorboard_scalars(run_dir)
    samples = sample_metrics(run_dir, boundary_period, detail_reference)

    losses = [
        finite_float(entry.get("loss"))
        for entry in entries
        if finite_float(entry.get("loss")) is not None
    ]
    final = entries[-1] if entries else {}
    row = dict(
        architecture=name,
        status=status_for_run(run_dir),
        run_dir=str(run_dir),
        loaded_tensors=compatibility["loaded"],
        skipped_tensors=compatibility["skipped"],
        missing_tensors=compatibility["missing"],
        parameters=compatibility["parameters"],
        final_iteration=final.get("iter"),
        final_loss=finite_float(final.get("loss")),
        minimum_logged_loss=min(losses) if losses else None,
        final_grad_norm=finite_float(final.get("grad_norm")),
        final_it_per_second=finite_float(final.get("it_s")),
        loss_auxiliary=tensorboard.get("train/loss_auxiliary"),
        loss_gm_mean=tensorboard.get("train/loss_gm_mean"),
        loss_voxel_gradient=tensorboard.get("train/loss_voxel_gradient"),
    )
    row.update(samples)
    return row


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(rows, path):
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_markdown(rows, path, detail_reference, boundary_period):
    columns = [
        ("architecture", "Architecture"),
        ("status", "Status"),
        ("loaded_tensors", "Loaded"),
        ("skipped_tensors", "Skipped"),
        ("final_loss", "Final loss"),
        ("final_grad_norm", "Grad norm"),
        ("boundary_ratio", "Boundary ratio"),
        ("detail_to_lll", "Detail/LLL"),
        ("age_l1_delta", "Age L1 delta"),
        ("nonfinite_fraction", "Nonfinite"),
    ]
    lines = [
        "# GMFlow3D Architecture Comparison",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join(" --- " for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(key)) for key, _ in columns)
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Patch-boundary period: {boundary_period} voxels. Values closer to 1 are better.",
        f"- Real-data Haar detail/LLL reference: {detail_reference:.4f}.",
        "- Age L1 delta measures response strength under fixed noise; larger is not automatically better.",
        "- Final losses are directly comparable only when the objective weights are identical.",
        "- These proxy metrics do not replace anatomical, segmentation, or clinical validation.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compare GMFlow3D architecture-ablation runs")
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True,
        help="Run mapping in NAME=PATH format; repeat for each architecture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-period", type=int, default=4)
    parser.add_argument("--detail-reference", type=float, default=0.0884)
    args = parser.parse_args()

    if args.boundary_period < 2:
        parser.error("--boundary-period must be at least 2")

    rows = [
        summarize_run(
            name, path, args.boundary_period, args.detail_reference)
        for name, path in args.run
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    markdown_path = args.output_dir / "comparison.md"
    json_path = args.output_dir / "comparison.json"

    write_csv(rows, csv_path)
    write_markdown(
        rows, markdown_path, args.detail_reference, args.boundary_period)
    json_path.write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    for row in rows:
        print(
            f"{row['architecture']}: status={row['status']} "
            f"boundary={format_value(row['boundary_ratio'])} "
            f"detail={format_value(row['detail_to_lll'])} "
            f"age_delta={format_value(row['age_l1_delta'])}")


if __name__ == "__main__":
    main()
