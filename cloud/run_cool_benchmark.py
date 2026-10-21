#!/usr/bin/env python3
"""run_cool_benchmark.py  —  Phase 3 §3.2 item 4: COOL benchmarking suite.

Follows OpenCV's COOL-Benchmark guidance: time the core inspection kernels
(bilateral filter, adaptive threshold, morphology, connected components,
distance transform) over N frames, report per-kernel latency (ms), frame
throughput (FPS), CPU utilization, then project cloud cost per 10,000
frames for the standard vs COOL builds.

On Graviton the script imports the KleidiCV-accelerated cv2 from /opt/cool;
on the laptop it measures the local x86_64 build as the baseline leg.
`--emit expected` prints the §3.3 acceptance projection (1.8–2.0x) without
claiming a measurement; any real run reports measured numbers only.
"""

import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
scripts_dir = os.path.join(os.path.dirname(current_dir), 'src',
                           'robotic_4dof_arm', 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
import defect_inspector as di  # noqa: E402

GRAVITON_PER_HOUR_USD = 0.03456  # c7g.large on-demand (docs.aws.amazon.com)
X86_PER_HOUR_USD = 0.04160        # c6i.large on-demand (docs.aws.amazon.com)
FRAMES_PER_JOB = 10_000


def _cpu_percent():
    try:
        with open('/proc/stat') as fh:
            a = list(map(int, fh.readline().split()[1:8]))
        time.sleep(0.2)
        with open('/proc/stat') as fh:
            b = list(map(int, fh.readline().split()[1:8]))
        idle = b[3] - a[3]
        total = sum(b) - sum(a)
        return round(100.0 * (1.0 - idle / max(total, 1)), 1)
    except Exception:
        return None


def _synthetic_roi(seed=7, size=320):
    rng = np.random.default_rng(seed)
    img = rng.integers(150, 220, (size, size)).astype(np.uint8)
    cv2.line(img, (30, size // 2), (size - 30, size // 2), 30, 3)
    return img


def run_kernels(frames=50, size=320):
    """Time each core kernel; returns {kernel: ms_per_frame} + fps."""
    img = _synthetic_roi(size=size)
    kernels = {
        'bilateral': lambda: cv2.bilateralFilter(img, 7, 50.0, 7.0),
        'adaptive_threshold': lambda: cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 7.0),
        'morphology': lambda: cv2.morphologyEx(
            cv2.morphologyEx(img, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (3, 3))),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))),
        'connected_components': lambda: cv2.connectedComponentsWithStats(
            (img > 128).astype(np.uint8), 8),
        'distance_transform': lambda: cv2.distanceTransform(
            (img > 128).astype(np.uint8), cv2.DIST_L2, 3),
        'analyze_roi': lambda: di.analyze_roi(img, 462.0, 0.45),
    }
    lat = {}
    for name, fn in kernels.items():
        fn()  # warm-up (TBB pools, frequency ramp)
        t0 = time.perf_counter()
        for _ in range(frames):
            fn()
        lat[name] = round((time.perf_counter() - t0) / frames * 1000.0, 3)
    # End-to-end throughput from the full pipeline kernel.
    t0 = time.perf_counter()
    for _ in range(frames):
        kernels['analyze_roi']()
    fps = round(frames / (time.perf_counter() - t0), 2)
    return lat, fps


def cost_per_10k(fps, per_hour_usd):
    secs = FRAMES_PER_JOB / max(fps, 1e-9)
    return round(secs / 3600.0 * per_hour_usd, 4)


def main():
    ap = argparse.ArgumentParser(description='COOL benchmark suite')
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--label', default=None)
    ap.add_argument('--emit', choices=('measured', 'expected'),
                    default='measured')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    info = cv2.getBuildInformation()
    lat, fps = run_kernels(args.frames, args.size)
    result = {
        'label': args.label or platform.machine(),
        'arch': platform.machine(),
        'opencv': cv2.__version__,
        'kleidicv': bool('KleidiCV' in info
                         or os.environ.get('COOL_KLEIDICV', '')),
        'frames': args.frames, 'roi_size': args.size,
        'latency_ms_per_frame': lat, 'fps': fps,
        'cpu_percent': _cpu_percent(),
        'cost_usd_per_10k': {
            'x86_reference': cost_per_10k(fps, X86_PER_HOUR_USD),
            'graviton': cost_per_10k(fps, GRAVITON_PER_HOUR_USD)},
    }
    if args.emit == 'expected':
        result['projection'] = {
            'note': '§3.3 acceptance band — NOT a measurement',
            'speedup_vs_x86': [1.8, 2.0]}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(result, fh, indent=2)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
