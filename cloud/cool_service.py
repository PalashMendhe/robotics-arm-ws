"""cool_service.py  —  Phase 3 §3.2 item 2: edge-cloud ROI inspection API.

Runs inside Dockerfile.cool on the Graviton host.  The edge (laptop) keeps
robot control + video; it POSTs compressed ROI snapshots (<100 KB JPEG) here
for KleidiCV-accelerated profiling, and asks for ASTM-style certificates that
are sealed + archived to S3 (Phase 3 §3.2 item 3, audit_ledger.py).

  GET  /health          → build info (opencv version, KleidiCV flag, arch)
  POST /analyze         → multipart JPEG + fx/depth_m → defect metrics
  POST /certify         → {verdict, metrics, workpiece_serial} → sealed record

Pure stdlib + cv2/numpy/fastapi/boto3 — no ROS imports, so this module runs
identically on the laptop (tests) and on Graviton (production).
"""

import hashlib
import io
import json
import os
import platform
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
scripts_dir = os.path.join(os.path.dirname(current_dir), 'src',
                           'robotic_4dof_arm', 'scripts')
for _p in (current_dir, scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import defect_inspector as di

try:
    from audit_ledger import archive_certificate
except Exception:  # laptop without AWS creds still serves /analyze
    archive_certificate = None

MAX_ROI_BYTES = 100 * 1024  # §3.2 item 2: snapshots stay under 100 KB

app = FastAPI(title='ActiveInspect-5 COOL inspection service')


@app.get('/health')
def health():
    return {'ok': True,
            'opencv': cv2.__version__,
            'kleidicv': bool(cv2.getBuildInformation().count('KleidiCV')
                             or os.environ.get('COOL_KLEIDICV', '')),
            'arch': platform.machine(),
            's3_ledger': archive_certificate is not None}


@app.post('/analyze')
async def analyze(roi: UploadFile = File(...),
                  fx: float = Form(...),
                  depth_m: float = Form(...)):
    raw = await roi.read()
    if len(raw) > MAX_ROI_BYTES:
        raise HTTPException(413, f'ROI {len(raw)} B exceeds 100 KB budget')
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(400, 'ROI is not a decodable image')
    metrics = di.analyze_roi(img, float(fx), float(depth_m))
    return JSONResponse({'ok': True, 'roi_bytes': len(raw),
                         'metrics': metrics})


def _inspection_policy():
    """Confidence thresholds from robot_params.yaml (single source)."""
    params = os.path.join(os.path.dirname(current_dir), 'src',
                          'arm_moveit_config', 'config', 'robot_params.yaml')
    if not os.path.isfile(params):
        params = os.path.join(os.path.dirname(current_dir),
                              'robot_params.yaml')
    try:
        import yaml
        with open(params) as fh:
            insp = yaml.safe_load(fh)['inspection']
        return {'certify_threshold': float(insp['certify_threshold']),
                'triage_low': float(insp['triage_low'])}
    except Exception:
        return {'certify_threshold': 0.85, 'triage_low': 0.70}


def _seal(record):
    body = {k: v for k, v in record.items() if k != 'sha256'}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


class CertifyRequest(BaseModel):
    verdict: str
    metrics: dict
    workpiece_serial: str
    frames_b64: list = []


@app.post('/certify')
async def certify(req: CertifyRequest):
    insp = _inspection_policy()
    p = float(req.metrics.get('confidence', 0.0))
    if insp['triage_low'] < p < insp['certify_threshold']:
        return JSONResponse({'ok': False, 'escalated': True,
                             'reason': 'borderline confidence — triage',
                             'confidence': p})
    record = {'workpiece_serial': req.workpiece_serial,
              'verdict': req.verdict, 'metrics': req.metrics}
    digest = _seal(record)
    record['sha256'] = digest
    s3_keys: list = []
    if archive_certificate is not None:
        try:
            import base64
            frames = [base64.b64decode(f) for f in req.frames_b64]
            _, s3_keys = archive_certificate(
                req.workpiece_serial, req.verdict, req.metrics,
                frames=frames)
        except Exception as exc:
            raise HTTPException(502, f'S3 archive failed: {exc}')
    return JSONResponse({'ok': True, 'escalated': False,
                         'sha256': digest, 's3_keys': s3_keys,
                         'record': record})

