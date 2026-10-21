#!/usr/bin/env python3
"""simulate_pipeline.py — ActiveInspect-5 end-to-end offline simulation.

Drives the full MCP tool chain without ROS or AWS:
  survey_workpiece → inspect_viewpoint (oblique) → run_opencv_analysis
  → disambiguate → certify_inspection / triage

Runs two scenarios:
  A) Glare-masked structural crack  → confirmed CRACK after oblique view
  B) Benign oil stain               → confirmed BENIGN, no certificate issued

Prints a human-readable report and writes a JSON report to /tmp/activeinspect_report.json.
"""

import asyncio
import json
import os
import sys
import datetime

# ── Path setup (no sourced ROS workspace needed) ─────────────────────────────
SCRIPTS = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       'src', 'robotic_4dof_arm', 'scripts')
sys.path.insert(0, SCRIPTS)

import mcp_server as _srv
import defect_inspector as _di
import scene_config as _scene

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def _hdr(msg): print(f'\n{BOLD}{CYAN}{"─"*60}{RESET}');\
               print(f'{BOLD}{CYAN}  {msg}{RESET}');\
               print(f'{BOLD}{CYAN}{"─"*60}{RESET}')

def _ok(msg):   print(f'  {GREEN}✔  {msg}{RESET}')
def _warn(msg): print(f'  {YELLOW}⚠  {msg}{RESET}')
def _fail(msg): print(f'  {RED}✘  {msg}{RESET}')
def _info(msg): print(f'     {msg}')


async def run_scenario(name: str, roi_id: str, approach: str = 'front',
                       report: dict = None):
    """Run one full inspection scenario and return its result dict."""
    _hdr(f'Scenario: {name}')
    _srv._reset_state()

    # ── Step 1: Survey ────────────────────────────────────────────────────────
    print(f'\n{BOLD}[1/4] survey_workpiece(){RESET}')
    survey = await _srv.survey_workpiece()
    _ok(f'Backend: {survey["backend"]} | survey_height: {survey["survey_height_m"]} m')
    if not survey['candidates']:
        _fail('No candidates found — scenario aborted.')
        return None
    tid = survey['candidates'][0]['target_id']
    conf0 = survey['candidates'][0]['confidence']
    _ok(f'Candidate #{tid} | centroid: {survey["candidates"][0]["centroid_mm"]} mm | '
        f'confidence: {conf0:.0%}')
    _warn('Glare ambiguity detected — dispatching oblique viewpoint...')

    # ── Step 2: Oblique viewpoint ─────────────────────────────────────────────
    print(f'\n{BOLD}[2/4] inspect_viewpoint(target={tid}, angle={approach!r}, dist=120 mm){RESET}')
    vp = await _srv.inspect_viewpoint(tid, approach, 120.0)
    if not vp['ok']:
        _fail(f'Viewpoint refused: {vp["reason"]}')
        return None
    _ok(f'Arm moved (simulated): ik_ok={vp["ik_ok"]} | '
        f'viewpoints used: {vp["viewpoints_used"]}/{vp["viewpoints_used"] + vp["viewpoints_remaining"]}')
    _info(f'Oblique pose: x={vp["pose"]["x"]:.3f}  y={vp["pose"]["y"]:.3f}  z={vp["pose"]["z"]:.3f}')

    # ── Step 3: OpenCV analysis ───────────────────────────────────────────────
    print(f'\n{BOLD}[3/4] run_opencv_analysis(roi_id={roi_id!r}){RESET}')
    analysis = await _srv.run_opencv_analysis(roi_id)
    if not analysis['ok']:
        _fail(f'Analysis failed: {analysis["reason"]}')
        return None
    m = analysis['metrics']
    _ok(f'ROI id: {analysis["roi_id"]}')
    _info(f'  length      : {m["length_mm"]:.2f} mm')
    _info(f'  max width   : {m["max_width_mm"]:.2f} mm')
    _info(f'  aspect ratio: {m["aspect_ratio"]:.1f}')
    _info(f'  is_crack_like: {m["is_crack_like"]}')
    _info(f'  confidence  : {m["confidence"]:.0%}')
    _info(f'  mm/pixel    : {m["mm_per_pixel"]:.4f}')

    # ── Multi-view disambiguation ─────────────────────────────────────────────
    survey_m = dict(m, centroid_mm=[12.0, -4.0])
    oblique_m = dict(m, centroid_mm=[12.3, -3.8] if roi_id == 'synthetic-crack'
                     else [18.5, 2.1])   # stain shifts far → benign
    verdict, conf_final = _di.disambiguate(survey_m, oblique_m)
    _info(f'  disambiguate → verdict={verdict!r}  confidence={conf_final:.0%}')

    # ── Step 4: Certify ───────────────────────────────────────────────────────
    print(f'\n{BOLD}[4/4] certify_inspection(verdict={verdict!r}){RESET}')
    cert_metrics = dict(m, confidence=conf_final, centroid_mm=[12.0, -4.0])
    cert = await _srv.certify_inspection(verdict, cert_metrics)

    result = {
        'scenario': name,
        'roi_id': roi_id,
        'candidate_id': tid,
        'survey_confidence': conf0,
        'opencv_metrics': m,
        'verdict': verdict,
        'final_confidence': conf_final,
        'escalated': cert.get('escalated', False),
        'cert_sha256': cert.get('sha256'),
        'cert_path': cert.get('path'),
    }

    if cert.get('escalated'):
        _warn(f'ESCALATED to human triage '
              f'(confidence {conf_final:.0%} in borderline band '
              f'{cert["triage_band"][0]:.0%}–{cert["triage_band"][1]:.0%})')
    elif cert.get('ok'):
        _ok(f'Certificate issued → {cert["path"]}')
        _ok(f'SHA-256: {cert["sha256"][:32]}…')
    else:
        _fail(f'Certification failed: {cert}')

    if report is not None:
        report['scenarios'].append(result)

    return result


async def main():
    print(f'\n{BOLD}{"="*60}')
    print(f'  ActiveInspect-5 — Offline Pipeline Simulation')
    print(f'  {datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}')
    print(f'{"="*60}{RESET}')
    print(f'  INSPECTION policy loaded from: {_scene.load_path()}')
    print(f'  survey_height : {_scene.INSP_SURVEY_HEIGHT_M} m')
    print(f'  oblique_tilt  : {_scene.INSP_OBLIQUE_TILT_DEG}°')
    print(f'  max_viewpoints: {_scene.INSP_MAX_VIEWPOINTS}')
    print(f'  triage band   : {_scene.INSP_TRIAGE_LOW:.0%} – {_scene.INSP_CERTIFY_THRESHOLD:.0%}')

    report = {
        'generated_at': datetime.datetime.now().isoformat(),
        'policy': {
            'survey_height_m': _scene.INSP_SURVEY_HEIGHT_M,
            'oblique_tilt_deg': _scene.INSP_OBLIQUE_TILT_DEG,
            'max_viewpoints': _scene.INSP_MAX_VIEWPOINTS,
            'triage_band': [_scene.INSP_TRIAGE_LOW, _scene.INSP_CERTIFY_THRESHOLD],
        },
        'scenarios': [],
    }

    # ── Scenario A: Structural crack ──────────────────────────────────────────
    await run_scenario(
        name='Glare-masked structural crack',
        roi_id='synthetic-crack',
        approach='front',
        report=report,
    )

    # ── Scenario B: Benign oil stain ──────────────────────────────────────────
    await run_scenario(
        name='Benign oil stain (false-positive candidate)',
        roi_id='synthetic-stain',
        approach='left',
        report=report,
    )

    # ── Viewpoint budget enforcement demo ─────────────────────────────────────
    _hdr('Bounded Autonomy: viewpoint budget enforcement')
    _srv._reset_state()
    cid = _srv._register_simulated_candidate()
    for angle in ('front', 'left', 'right'):
        r = await _srv.inspect_viewpoint(cid, angle, 120.0)
        _ok(f'{angle:6s} → ok={r["ok"]}  used={r["viewpoints_used"]}  '
            f'remaining={r["viewpoints_remaining"]}')
    refused = await _srv.inspect_viewpoint(cid, 'back', 120.0)
    _fail(f'back   → ok={refused["ok"]}  reason: {refused["reason"]}')
    report['budget_enforcement'] = {'refused_on_4th_viewpoint': not refused['ok']}

    # ── Summary ───────────────────────────────────────────────────────────────
    _hdr('Pipeline Summary')
    for s in report['scenarios']:
        verdict_color = GREEN if s['verdict'] == 'crack' else \
                        YELLOW if s['verdict'] == 'inconclusive' else GREEN
        escalated_str = f'{YELLOW}(→ triage){RESET}' if s['escalated'] else ''
        print(f'  {verdict_color}{BOLD}{s["verdict"].upper():14s}{RESET}  '
              f'{s["scenario"]}  conf={s["final_confidence"]:.0%}  {escalated_str}')

    budget_ok = report['budget_enforcement']['refused_on_4th_viewpoint']
    print(f'  {GREEN if budget_ok else RED}{BOLD}BUDGET GUARD{"":6s}{RESET}  '
          f'4th viewpoint correctly {"refused" if budget_ok else "ALLOWED (BUG)"}')

    # ── Write JSON report ─────────────────────────────────────────────────────
    out_path = '/tmp/activeinspect_report.json'
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f'\n{BOLD}Report written → {out_path}{RESET}\n')


if __name__ == '__main__':
    asyncio.run(main())

