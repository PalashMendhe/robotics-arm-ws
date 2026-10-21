# src/robotic_4dof_arm/test/test_cloud_cool.py
"""Phase 3 tests: cloud service, S3 audit ledger, COOL benchmark.

No ROS, no AWS credentials, no docker: FastAPI TestClient drives
/analyze + /certify, botocore Stubber stands in for S3, and the benchmark
runs a 2-frame smoke leg.
"""
import os
import sys

import numpy as np

CLOUD = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                      '..', '..', '..', 'cloud'))
sys.path.insert(0, CLOUD)
SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, SCRIPTS)

import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import audit_ledger  # noqa: E402
import cool_service  # noqa: E402
from run_cool_benchmark import cost_per_10k, run_kernels  # noqa: E402

client = TestClient(cool_service.app)


def _crack_jpeg(size=320):
    img = np.full((size, size), 200, np.uint8)
    cv2.line(img, (30, size // 2), (size - 30, size // 2), 30, 3)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return bytes(buf)


def _metrics(confidence):
    return {'length_mm': 14.2, 'max_width_mm': 0.7, 'aspect_ratio': 20.3,
            'is_crack_like': True, 'confidence': confidence}


def test_health_reports_build():
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] and body['opencv'] and body['arch']


def test_analyze_crack_roi():
    payload = _crack_jpeg()
    assert len(payload) < 100 * 1024  # §3.2 item 2 budget
    r = client.post('/analyze', files={'roi': ('roi.jpg', payload)},
                    data={'fx': '462.0', 'depth_m': '0.45'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['ok'] and body['roi_bytes'] == len(payload)
    assert body['metrics']['is_crack_like']


def test_analyze_rejects_oversize_roi():
    big = b'\xff' * (100 * 1024 + 1)
    r = client.post('/analyze', files={'roi': ('roi.jpg', big)},
                    data={'fx': '462.0', 'depth_m': '0.45'})
    assert r.status_code == 413


def test_analyze_rejects_undecodable_roi():
    r = client.post('/analyze', files={'roi': ('roi.jpg', b'not-an-image')},
                    data={'fx': '462.0', 'depth_m': '0.45'})
    assert r.status_code == 400


def test_certify_issues_sealed_record(monkeypatch):
    monkeypatch.setattr(cool_service, 'archive_certificate', None)
    body = {'verdict': 'crack', 'metrics': _metrics(0.92),
            'workpiece_serial': 'WP-001'}
    r = client.post('/certify', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['ok'] and not out['escalated']
    assert len(out['sha256']) == 64
    assert out['record']['sha256'] == out['sha256']


def test_certify_borderline_escalates(monkeypatch):
    monkeypatch.setattr(cool_service, 'archive_certificate', None)
    body = {'verdict': 'crack', 'metrics': _metrics(0.78),
            'workpiece_serial': 'WP-002'}
    r = client.post('/certify', json=body)
    assert r.status_code == 200
    assert r.json()['escalated'] is True  # triage band 0.70-0.85


def test_certify_archives_frames_to_s3(monkeypatch):
    import base64 as _b64
    frame = _b64.b64encode(_crack_jpeg()).decode()
    seen = {}

    def fake_archive(serial, verdict, metrics, frames=(), **kw):
        seen.update(serial=serial, n_frames=len(frames))
        return 'digest123', [f'{serial}/record.json']

    monkeypatch.setattr(cool_service, 'archive_certificate', fake_archive)
    body = {'verdict': 'crack', 'metrics': _metrics(0.95),
            'workpiece_serial': 'WP-003', 'frames_b64': [frame]}
    r = client.post('/certify', json=body)
    assert r.status_code == 200
    assert r.json()['s3_keys'] == ['WP-003/record.json']
    assert seen == {'serial': 'WP-003', 'n_frames': 1}


def test_ledger_bundle_layout_and_seal():
    import json as _json
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 255
    digest, bundle = audit_ledger.build_bundle(
        'WP-007', 'crack', _metrics(0.9),
        frames=[_crack_jpeg(size=64)], masks=[mask])
    assert len(digest) == 64
    keys = sorted(bundle)
    assert any(k.endswith('record.json') for k in keys)
    assert any(k.endswith('frame_00.jpg') for k in keys)
    assert any(k.endswith('mask_00.png') for k in keys)
    rec_key = [k for k in keys if k.endswith('record.json')][0]
    assert _json.loads(bundle[rec_key])['sha256'] == digest


def test_ledger_archive_uses_stubbed_s3():
    import boto3
    from botocore.stub import ANY, Stubber
    s3 = boto3.client('s3', region_name='us-east-1',
                      aws_access_key_id='t', aws_secret_access_key='t')
    stub = Stubber(s3)
    digest, bundle = audit_ledger.build_bundle('WP-008', 'benign',
                                               _metrics(0.2))
    for key in sorted(bundle):
        short = key.split('activeinspect-audit/', 1)[1]
        stub.add_response('put_object', {},
                          {'Bucket': 'activeinspect-audit', 'Key': short,
                           'Body': ANY})
    stub.activate()
    out_digest, keys = audit_ledger.archive_certificate(
        'WP-008', 'benign', _metrics(0.2), s3_client=s3)
    stub.deactivate()
    assert out_digest == digest and len(keys) == len(bundle)


def test_benchmark_kernels_smoke():
    lat, fps = run_kernels(frames=2, size=64)
    for kernel in ('bilateral', 'adaptive_threshold', 'morphology',
                   'connected_components', 'distance_transform',
                   'analyze_roi'):
        assert lat[kernel] >= 0
    assert fps > 0
    assert cost_per_10k(fps, 0.04) > 0


def test_cool_dockerfile_targets_arm64():
    path = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                         '..', '..', '..',
                                         'Dockerfile.cool'))
    text = open(path).read()
    assert 'linux/arm64' in text and 'KleidiCV' in text
    assert 'cool_service' in text

