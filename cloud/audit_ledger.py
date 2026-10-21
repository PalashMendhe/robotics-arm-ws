"""audit_ledger.py  —  Phase 3 §3.2 item 3: S3 cryptographic audit ledger.

Bundles each certified inspection (multi-angle frames, contour mask PNGs,
JSON metric metadata + SHA-256 seal) into
s3://<bucket>/<workpiece_serial>/<sha256>/ and returns the key manifest.

boto3 client is injectable so pytest covers the bundle layout + seal with
botocore Stubber — no AWS credentials needed for the test suite.
"""

import hashlib
import io
import json

import boto3
import cv2
import numpy as np

LEDGER_PREFIX = 'activeinspect-audit'


def sha256_seal(record):
    """Canonical SHA-256 over the sorted-JSON record (no seal field)."""
    body = {k: v for k, v in record.items() if k != 'sha256'}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def encode_png(gray):
    """Grayscale ndarray → PNG bytes (contour masks, §3.2 item 3)."""
    ok, buf = cv2.imencode('.png', np.ascontiguousarray(gray))
    if not ok:
        raise ValueError('PNG encode failed')
    return bytes(buf)


def build_bundle(workpiece_serial, verdict, metrics, frames=(),
                 masks=()):
    """Assemble the ledger bundle; returns (digest, {key: bytes})."""
    record = {'workpiece_serial': workpiece_serial,
              'verdict': verdict, 'metrics': metrics}
    digest = sha256_seal(record)
    record['sha256'] = digest
    base = f'{LEDGER_PREFIX}/{workpiece_serial}/{digest}'
    bundle = {f'{base}/record.json':
              json.dumps(record, indent=2, sort_keys=True).encode()}
    for i, frame in enumerate(frames):
        bundle[f'{base}/frame_{i:02d}.jpg'] = bytes(frame)
    for i, mask in enumerate(masks):
        bundle[f'{base}/mask_{i:02d}.png'] = encode_png(mask)
    return digest, bundle


def archive_certificate(workpiece_serial, verdict, metrics, frames=(),
                        masks=(), bucket='activeinspect-audit',
                        s3_client=None):
    """Upload a sealed bundle; returns (digest, [keys])."""
    s3 = s3_client or boto3.client('s3')
    digest, bundle = build_bundle(workpiece_serial, verdict, metrics,
                                  frames, masks)
    keys = []
    for key, payload in bundle.items():
        short = key.split(f'{LEDGER_PREFIX}/', 1)[1]
        s3.put_object(Bucket=bucket, Key=short, Body=io.BytesIO(payload))
        keys.append(short)
    return digest, keys
