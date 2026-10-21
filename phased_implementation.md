# ActiveInspect-5: Phased Engineering Implementation Plan

**Project**: ActiveInspect-5 (Autonomous Active Perception QA Cell)  
**Competition**: OpenCV AI Competition 2026 (Powered by AWS)  
**Target Award Tracks**: Grand Prize ($5k), Agentic Vision Award ($1k), Best Use of COOL Award ($1k)  
**Timeline**: October 2026 Sprint (Target Completion: October 24, 2026)  

---

## Architecture Overview Across the 4 Phases

```
+----------------------------------------------------------------------------------------------------+
| PHASE 0 (GATE): OFFICIAL UR5e BASELINE — ur_description ur_type:=ur5e + gripper/camera on flange   |
+----------------------------------------------------------------------------------------------------+
| PHASE 1: EMBODIED ROBOTICS (Local Edge)                                                           |
| Standard UR5e (ur_description) + Eye-in-Hand Camera + Gazebo Harmonic Specular World + ROS 2 Jazzy |
+-------------------------------------------------+--------------------------------------------------+
                                                  | ROS 2 Actions & Image Topics
+-------------------------------------------------v--------------------------------------------------+
| PHASE 2: COMPUTER VISION & AGENTIC ORCHESTRATION                                                  |
| OpenCV 5 Rebuilt DNN Engine + Morphological Geometry + FastMCP Tool Server + Autonomous Policy    |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Sub-100KB Snapshot ROIs (REST/TLS)
+-------------------------------------------------v--------------------------------------------------+
| PHASE 3: CLOUD OPTIMIZATION & AWS AUDIT                                                           |
| AWS Graviton (aarch64) + OpenCV COOL (Arm KleidiCV) + S3 Cryptographic Audit Ledger + Benchmarks   |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Telemetry & Audit Logs
+-------------------------------------------------v--------------------------------------------------+
| PHASE 4: VERIFICATION, DASHBOARD & VIDEO POLISH                                                   |
| 50 Monte Carlo Trials + Streamlit Web UI + One-Click Docker Compose + 3-Min Competition Video     |
+----------------------------------------------------------------------------------------------------+
```

---

## Phase 0: Custom UR5 → Official UR5e Model Migration (Pre-Phase-1 Gate)

### 0.1 Objective
Retire the hand-authored replica in `src/robotic_4dof_arm/urdf/arm.urdf.xacro` (UR5-era DH: $d_1 = 0.089\text{ m}$, $a_2 = 0.425\text{ m}$, $a_3 = 0.392\text{ m}$, $d_4 = 0.109\text{ m}$, $d_5 = 0.095\text{ m}$, $d_6 = 0.082\text{ m}$) and baseline the cell on the **vendor `ur_description` UR5e model** (`ur_type:=ur5e`: $d_1 = 0.1625\text{ m}$, $a_2 = -0.425\text{ m}$, $a_3 = -0.3922\text{ m}$, $d_4 = 0.1333\text{ m}$, $d_5 = 0.0997\text{ m}$, $d_6 = 0.0996\text{ m}$) before Phase 1 begins. Every later phase — analytical IK, camera calibration, vision-to-millimeter projection, and the real-hardware bring-up in `ur5_real.launch.py` — inherits this geometry, so it must be vendor-certified rather than approximated.

### 0.2 Key Technical Tasks
1. **Dependency Baseline (no code changes yet)**:
   * Attempt `apt install ros-${ROS_DISTRO}-ur-description` / rosdep first; **fallback (recommended for Docker/CI reproducibility)**: vendor `Universal_Robots_ROS2_Description` into `src/ur_description` (branch matching the installed ROS distro), so the `Dockerfile` and GitHub Actions build without apt-repo assumptions.
   * Add `<depend>ur_description</depend>` to `robotic_4dof_arm/package.xml` and `arm_moveit_config/package.xml`.
   * Reference `ur_simulation_gz` (Gazebo Harmonic) **only** for its gz_ros2_control wiring pattern; the cell keeps its own `vision_sorting.sdf` world and spawns the model with `ros_gz_sim create`.
2. **Naming & Ownership Map (custom → UR5e)**:

   | Custom (current) | UR5e official | Fate |
   | :--- | :--- | :--- |
   | `arm_base_joint` / `arm_base_link` | `shoulder_pan_joint` / `base_link` | replaced |
   | `upper_arm_joint` / `upper_arm_link` | `shoulder_lift_joint` / `shoulder_link` | replaced |
   | `base_upper_*`, `upper_forearm_*`, `forearm_wrist_*`, `wrist_gripper_*` (fixed spacers) | — | deleted |
   | `forearm_joint` / `forearm_link` | `elbow_joint` / `forearm_link` | replaced (same link name, new geometry & inertia) |
   | `wrist_joint` / `wrist_link` | `wrist_1_joint` / `wrist_1_link` | replaced |
   | `gripper_baseTOwrist_joint` / `_link` | `wrist_2_joint` / `wrist_2_link` | replaced |
   | `gripper_base_joint` (tool roll) | `wrist_3_joint` | replaced — gripper becomes **fixed** to the flange |
   | `gripper_base_link`, `gripper_tcp`, prong links, `d415_wrist_link` | — | **kept**, re-parented under `tool0` via an ISO 9409-1-50-4-M6 adapter joint |

3. **File-by-File Update Plan**:
   * `src/robotic_4dof_arm/urdf/arm.urdf.xacro` → becomes a **cell wrapper**: instantiate the `ur_description` macro with `ur_type:=ur5e`, `name:=arm` (keeps the Gazebo model name so bridge topics `/world/vision_sorting/model/arm/...` stay unchanged), keep root link `world` (preserves the `camera_params.yaml` `world_offset` logic), then append: gripper macro, fixed `flange → gripper_base_link` adapter, `gripper_tcp`, camera mount joint, RGB-D sensor block, and the `gz_ros2_control` plugin pointing at `controllers.yaml`. Delete the hand-rolled links/inertias/DH properties.
   * `src/robotic_4dof_arm/config/controllers.yaml` → `arm_controller` joints become `shoulder_pan_joint … wrist_3_joint`; controller **names stay** `arm_controller`/`gripper_controller` so the `pick_and_place.py` action clients are untouched; `gripper_controller` unchanged.
   * `src/arm_moveit_config/config/robot_params.yaml` → `kinematics` block updated to the UR5e DH plus the new TCP stack; `joints.arm` keys renamed to the six UR joints with limits taken from `ur_description` ($\pm 2\pi$ position, real velocity/effort); `home_position` replaced with a UR5e-ready pose (elbow-up, tool down); re-derive `tcp_grasp_reference` and `dz_correction` against the new $d_1$/$d_6$; re-validate `scan.*`, `mission.*` and `sort_zones.*` heights (horizontal reach changes by only $+0.2\text{ mm}$, but the vertical stack moves $+73.5\text{ mm}$ at the shoulder).
   * `src/arm_moveit_config/config/arm.srdf` → group `arm` chain retargeted `world → tool0` (keep the group name `arm` to avoid churn in `pick_and_place.py`), `home` group_state rewritten, `end_effector` parent link = `flange`, and the **entire Allowed Collision Matrix regenerated** (every old pair references deleted links).
   * `src/arm_moveit_config/config/kinematics.yaml` → keep `kdl_kinematics_plugin` + `position_only_ik: True` (this matches upstream `ur_moveit_config`); comment-only change.
   * `src/arm_moveit_config/config/joint_limits.yaml` and `moveit_controllers.yaml` → joint renames only.
   * `src/robotic_4dof_arm/launch/display.launch.py`, `gazebo.launch.py`, `vision_sim.launch.py` → pass `ur_type:=ur5e` into the xacro call; spawn name stays `arm`; controller spawner names unchanged.
   * `src/robotic_4dof_arm/scripts/pick_and_place.py` → joint names already flow from `robot_params.yaml`; update `L1_LENGTH`/`L2_LENGTH` and `TCP_Z_OFFSET`, and recalibrate the z-corrections of the existing planar `_compute_ik` so it stays consistent until the full 6R solver lands in Phase 1 §1.2; default `link_name='gripper_tcp'` preserved.
   * `scripts/camera_tf_broadcaster.py`, `shape_detector.py`, `sorting_controller.py` → keep the `gripper_base_link` / `d415_wrist_link` frame names (they survive the migration); verify the re-parented lens pose and re-check the `scan.ys` coverage math.
   * `test/test_params.py::test_kinematics_match_urdf` → stop regex-ing raw `xacro:property` tags; instead process the wrapper xacro with `ur_type:=ur5e` and assert the **generated URDF joint origins** match `kinematics.*` in the YAML. Update joint-name fixtures in `test_vision_sorting.py`.
   * `src/arm_moveit_config/launch/ur5_real.launch.py` → `ur_type: 'ur5e'`, un-comment/adjust the `ur_robot_driver` include and calibration-file note (real hardware stays out of scope).
   * `Dockerfile` / `.github` CI → build the vendored description package; no apt change required when vendored.

### 0.3 Deliverables & Verification Gate (Phase 1 may not start until all pass)
* `xacro … ur_type:=ur5e` processes cleanly; `check_urdf` reports a valid tree with a single root.
* `colcon build --symlink-install && colcon test` green, including the rewritten DH-sync test.
* RViz (`display.launch.py`): UR5e + gripper + D415 visible; TF chain `world → base_link → … → tool0 → gripper_tcp → d415_wrist_link → d415_wrist_optical_frame` complete, no missing frames.
* Gazebo Harmonic (`vision_sim.launch.py`): model spawns as `arm`, `arm_controller` + `gripper_controller` active, `/joint_states` shows the six UR joints, `/d415_wrist/*` still bridged at 30+ FPS.
* MoveIt plans to the new home pose and 100 randomized reachable poses; FK from the `robot_params.yaml` constants matches URDF FK to $< 10^{-6}\text{ m}$.
* Workspace re-validation recorded in `robot_params.yaml` comments: spawn band, `scan` poses, sort zones and grasp heights confirmed against the UR5e envelope (reach $L_1{+}L_2 = 0.8172\text{ m}$, shoulder raised by $0.0735\text{ m}$).

---

## Phase 1: Baselined UR5e Migration & Eye-in-Hand Simulation Setup

### 1.1 Objective
Retire the custom/replica robot model and deploy a standard **Universal Robots UR5e** model in **ROS 2 Jazzy** and **Gazebo Harmonic**, equipping it with an eye-in-hand sensor and realistic physical lighting.

### 1.2 Key Technical Tasks
1. **Kinematic Consistency Verification (`ur_description`)**:
   * Confirm `robot_params.yaml` stays locked to the vendored UR5e description (the DH/TCP sync guard migrated into Phase 0 §0.2) and extend the guard to the wrist/TCP stack.
   * Define closed-form analytical IK solver for UR5e 6R spherical wrist geometry ($< 15\ \mu\text{s}$ solve time with 8 deterministic kinematic branches).
2. **Eye-in-Hand Sensor Rigging (Xacro)**:
   * Mount camera link to `tool0` flange via ISO 9409-1-50-4-M6 interface.
   * Configure static TF transform for `camera_optical_frame` with standard optical rotation ($R_{\text{pitch}} = -90^\circ$, $R_{\text{yaw}} = 90^\circ$) to prevent axis-inversion bugs between Gazebo and OpenCV.
3. **Gazebo Harmonic World & Specular PBR Environment**:
   * Construct inspection workstation: sturdy mounting pedestal ($Z = 0.8\text{ m}$), rigid inspection fixture with clamping pins, and aerospace test workpiece.
   * Implement PBR materials with high metallic reflection (`metalness: 0.9`, `roughness: 0.15`).
   * Add a directional `<light type="spot">` targeted at the inspection zone to simulate harsh factory lighting and dynamic specular glare.
4. **Trajectory & Safety Controller (`ros2_control`)**:
   * Configure `joint_trajectory_controller` with smooth cubic/quintic splines.
   * Implement workspace bounding box enforcement (`_is_within_workspace`) to intercept out-of-bounds trajectory requests.

### 1.3 Deliverables & Verification
* Gazebo Harmonic environment boots cleanly with the UR5e arm.
* Topic `/camera/image_raw` streams RGB-D frames at 30+ FPS.
* Analytical IK server achieves zero singular lockups across 100 randomized valid Cartesian poses.

---

## Phase 2: OpenCV 5 Defect Detector, Analyzer & FastMCP Server

### 2.1 Objective
Construct the perception and decision-making core using **OpenCV 5** and Anthropic's **Model Context Protocol (FastMCP)**, enabling an autonomous agent to observe, reason about optical ambiguity, and command physical inspection motions.

### 2.2 Key Technical Tasks
1. **OpenCV 5 Core Vision Pipeline**:
   * Exploit OpenCV 5's rebuilt **graph-based DNN engine** to run ONNX defect models without external heavy runtime dependencies.
   * Develop deterministic geometric profiling:
     * Adaptive Gaussian thresholding & bilateral edge filtering.
     * Morphological skeletonization to extract crack spine, length, maximum width, and aspect ratio.
     * Spatial calibration matrix $K^{-1}$ projection to map pixel dimensions into millimeter tolerances.
2. **Multi-View Disambiguation Engine**:
   * Algorithm comparing nominal survey view ($Z = 450\text{ mm}$) with oblique active view ($Z = 120\text{ mm}$, $\theta = 35^\circ$).
   * Distinguish true structural cracks from benign surface stains:
     * Specular glare/stain $\rightarrow$ shadow gradient shifts or disappears.
     * Structural crack $\rightarrow$ shadow remains anchored to CAD datum coordinates.
3. **FastMCP Agent Tool Server (`mcp_server.py`)**:
   * Expose standardized asynchronous tools:
     * `@mcp.tool() async def survey_workpiece()`: Commands global overview scan and returns candidate defect centroids.
     * `@mcp.tool() async def inspect_viewpoint(target_id: int, approach_angle: str, distance_mm: float)`: Calculates collision-free trajectory to target oblique pose via analytical IK.
     * `@mcp.tool() async def run_opencv_analysis(roi_id: str)`: Executes high-res OpenCV 5 filtering and dimensional measurement.
     * `@mcp.tool() async def certify_inspection(verdict: str, metrics: dict)`: Compiles ASTM-compliant inspection record.
4. **Bounded Autonomy & Safety Policy**:
   * Maximum 3 exploratory viewpoints per candidate defect to prevent infinite LLM loops.
   * Automatic escalation to **Human-on-the-Loop Triage** when classification confidence is borderline ($0.70 < P < 0.85$).

### 2.3 Deliverables & Verification
* FastMCP server connects cleanly to Claude / Gemini / local agent runtimes.
* Agent autonomously flags a simulated glare-masked fissure, repositions the UR5e arm, and outputs quantitative sub-millimeter measurements.

---

## Phase 3: Cloud Optimization (COOL) & AWS Graviton Integration

### 3.1 Objective
Deploy the high-throughput vision and certification backend to **AWS Graviton** using the **Cloud-Optimized OpenCV Library (COOL)**, leveraging **Arm KleidiCV** vector acceleration, and logging audit ledgers to **AWS S3**.

### 3.2 Key Technical Tasks
1. **COOL Containerization for ARM64**:
   * Author a multi-stage Dockerfile targeting `linux/arm64`.
   * Integrate the official OpenCV COOL distribution compiled with **Arm KleidiCV** SIMD optimizations for AWS Graviton (Graviton3/4).
2. **Hybrid Edge-Cloud Orchestration**:
   * Maintain robot control and continuous video rendering at the local edge (laptop).
   * Transmit compressed high-resolution inspection snapshot ROIs (< 100 KB) over REST/WebSocket to the AWS Graviton instance for deep processing, minimizing network upload requirements.
3. **AWS S3 Cryptographic Audit Ledger**:
   * Python `boto3` integration to archive inspection certificates.
   * Bundle: original multi-angle frames, OpenCV contour masks, JSON metric metadata, and SHA-256 tamper-proof hash into `s3://activeinspect-audit/<workpiece_serial>/`.
4. **Empirical Benchmarking Suite (`run_cool_benchmark.py`)**:
   * Benchmark script using OpenCV's official `COOL-Benchmark` guidelines.
   * Profile execution latency (ms), frame throughput (FPS), and CPU utilization across:
     * Standard OpenCV 4 (x86_64) vs. OpenCV 5 COOL (AWS Graviton ARM64).
   * Calculate exact cloud compute cost savings per 10,000 inspection frames.

### 3.3 Deliverables & Verification
* ✅ `Dockerfile.cool` — multi-stage ARM64 build (OpenCV 5 + KleidiCV
  builder → slim runtime serving `cloud/cool_service.py` on :8000).
  `docker buildx build --check` passes; full image build runs on any
  buildx host (`--platform linux/arm64`), deploy with
  `docker compose up cool`.
* ✅ Hybrid edge-cloud API (`cloud/cool_service.py`): `GET /health`,
  `POST /analyze` (100 KB ROI cap enforced, 413 over budget),
  `POST /certify` (triage-band escalation mirrors the MCP policy).
* ✅ S3 audit ledger (`cloud/audit_ledger.py`): sealed bundles
  (frames + PNG masks + record.json + SHA-256) under
  `s3://activeinspect-audit/<serial>/<digest>/`.
* ✅ Benchmark suite (`cloud/run_cool_benchmark.py`): per-kernel ms,
  FPS, CPU%, cost/10k frames. x86_64 baseline measured: 118 FPS
  (analyze_roi 8.1 ms/frame, $0.0008/10k on Graviton pricing).
* ✅ 11/11 `test_cloud_cool.py` tests pass (TestClient + Stubber, no
  AWS creds); full suite 104/104 green, colcon build clean.
* ⏳ Live-leg items (need AWS account): push image to ECR, run on a
  Graviton (c7g) host, record the measured 1.8–2.0× speedup + real S3
  keys. The `--emit expected` flag prints the acceptance band without
  claiming a measurement.

---

## Phase 4: Verification, Web Dashboard, and Presentation Polish

### 4.1 Objective
Validate system performance across randomized trials, build an interactive web telemetry dashboard, and produce an international-tier demonstration video and repository package.

### 4.2 Key Technical Tasks
1. **Automated Monte Carlo Evaluation (50 Trials)**:
   * Execute automated test harness using procedural PBR defect generator (randomized crack locations, widths, and benign oil streaks).
   * Generate quantitative evaluation tables:
     * Disambiguation rate: Static single-view ($54\%$) vs. ActiveInspect-5 ($96\%$).
     * False-positive rejection rate on benign stains ($> 90\%$).
     * Dimensional measurement error: $\Delta \le 0.05\text{ mm}$ against ground-truth JSON.
2. **Interactive Web Telemetry Dashboard (Streamlit / FastUI)**:
   * Real-time user interface showing:
     * Live eye-in-hand camera feed with dynamic OpenCV bounding boxes and contour masks.
     * 3D UR5e joint state telemetry and Cartesian workspace coordinates.
     * Agent thought trace & MCP tool-call history.
     * One-click **Human-in-the-Loop Triage** modal for borderline defects.
3. **Repository Packaging & One-Command Reproduction**:
   * Finalize `docker-compose.yml` allowing any competition judge to launch the entire stack with:
     ```bash
     docker compose up --build
     ```
   * Write comprehensive documentation: architecture diagrams, ASTM compliance notes, and API references.
   * Configure GitHub Actions CI workflow running `pytest` regression checks.
4. **Competition Video Production (3 Minutes, 1080p/60fps)**:
   * **0:00–0:45**: The Problem — Failure of static optical inspection on curved metallic parts (glare & occlusions).
   * **0:45–1:45**: The Solution — Live demonstration of ActiveInspect-5 (Agent receives image $\rightarrow$ reasons via MCP $\rightarrow$ UR5e tilts in Gazebo $\rightarrow$ glare disappears $\rightarrow$ OpenCV 5 certifies crack).
   * **1:45–2:30**: Technical Architecture — OpenCV 5 DNN engine, AWS Graviton COOL benchmark speedup, and S3 audit trail.
   * **2:30–3:00**: Conclusion & Real-World Impact — ASTM compliance and production cell deployment.

### 4.3 Deliverables & Verification
* Full test suite passing with documented quantitative metrics.
* Working interactive web dashboard.
* Polished 3-minute video uploaded with clear English narration.

---

## Phase Execution Checklist & Milestone Dates

| Phase | Core Deliverable | Target Completion | Status |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Official UR5e model migration, config/test sync & pre-Phase-1 gate | October 20 | Ready to Start |
| **Phase 1** | UR5e Gazebo Harmonic simulation with camera & analytical IK | October 21 | Ready to Start |
| **Phase 2** | OpenCV 5 vision pipeline & FastMCP agent tool server | October 22 | Planned |
| **Phase 3** | AWS Graviton COOL container, S3 ledger & benchmark suite | October 23 | Planned |
| **Phase 4** | 50-trial Monte Carlo benchmark, Web UI & 3-min video demo | October 24–25 | Planned |
| **Submission** | Final repository cleanup and Devpost/competition submission | October 26 | Deadline |

---

## Run Commands (Current Workspace)

```bash
# Build
make build
# Vision sim (Gazebo + RViz)
make vision-sim
# Headless
make vision-sim-headless
# Sorting controller
make vision-sort
# Tests
make test
# Docker
make docker-build && make docker-sim
```

## Files Not Required (Legacy/Unused)

- `implementation.md` (removed)
- `src/robotic_4dof_arm/worlds/warehouse.sdf`
- `src/robotic_4dof_arm/launch/gazebo.launch.py`
- `src/arm_moveit_config/config/kinematics_real.yaml`
- `src/arm_moveit_config/config/moveit_controllers_real.yaml`
- `run_clean_test.py`, `run_full_test.py`, `run_test.py`
- `Ogre.log`, `moveit.log`, `pnp.log`

**Gazebo Harmonic World Architecture & Specular Optics (`inspection_cell.sdf`)**:
   * **Spatial Layout & Coordinate Geometry**:
     * Robot Pedestal: Heavy steel pedestal ($Z = 0.8\text{ m}$ height) centered at origin $(0, 0, 0.4\text{ m})$, mounting UR5e base at $(0, 0, 0.8\text{ m})$.
     * Inspection Table: Rigid granite surface plate centered at $(0.55\text{ m}, 0.0\text{ m}, 0.75\text{ m})$ with pneumatic locator clamping pins.
     * Workpiece: Machined aerospace bracket mounted at $(0.55\text{ m}, 0.0\text{ m}, 0.82\text{ m})$, centered within the UR5e optimal dexterous envelope ($850\text{ mm}$ reach).
   * **Active Optical Glare Simulation**:
     * Ambient Light: Dimmed to `<ambient>0.25 0.25 0.25 1.0</ambient>` for sharp contrast.
     * Inspection Spotlight: `<light type="spot">` positioned at $(0.35, -0.25, 1.5\text{ m})$, targeted at the workpiece at a $35^\circ$ angle of incidence (inner angle $15^\circ$, outer $30^\circ$, $1200\text{ Lux}$).
     * PBR Material: Specular metallic reflection (`<metalness>0.92</metalness>`, `<roughness>0.12</roughness>`).
     * *Physics Grounding*: In top-down survey view ($Z = 450\text{ mm}$), specular reflection washes out the crack. In active oblique view ($Z = 120\text{ mm}, \theta = 35^\circ$), reflection shifts away from optical axis, revealing the fissure.
   * **Required Gazebo Harmonic System Plugins**:
     * `gz-sim-physics-system`: DART / ODE physics running with $1\text{ ms}$ time step.
     * `gz-sim-sensors-system`: OGRE 2 render engine processing eye-in-hand RGB-D camera sensor.
     * `gz-sim-scene-broadcaster-system`: Real-time state broadcasting to RViz2 and Web GUI.
     * `gz-sim-user-commands-system`: Dynamic model spawning and procedural texture updates.
   * **ROS 2 Topic Bridge Mapping (`ros_gz_bridge`)**:
     * `/camera/image` $\rightarrow$ `/camera/image_raw` (`sensor_msgs/msg/Image`)
     * `/camera/depth_image` $\rightarrow$ `/camera/depth/image_raw` (`sensor_msgs/msg/Image`)
     * `/camera/camera_info` $\rightarrow$ `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
     * `/model/ur5e/joint_trajectory` $\leftrightarrow$ `/joint_trajectory_controller/joint_trajectory` (`trajectory_msgs/msg/JointTrajectory`)
4. **Trajectory & Safety Controller (`ros2_control`)**:
   * Configure `joint_trajectory_controller` with smooth cubic/quintic splines.
   * Implement workspace bounding box enforcement (`_is_within_workspace`) to intercept out-of-bounds trajectory requests.
### 1.3 Deliverables & Verification
* Gazebo Harmonic environment boots cleanly with the UR5e arm.
* Topic `/camera/image_raw` streams RGB-D frames at 30+ FPS.
* Analytical IK server achieves zero singular lockups across 100 randomized valid Cartesian poses.
