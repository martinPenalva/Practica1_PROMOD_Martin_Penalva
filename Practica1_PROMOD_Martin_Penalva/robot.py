#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo procedural del Franka Emika Panda para la planta industrial MuJoCo.

IMPORTANTE:
- No usa POO.
- No define clases.
- Toda la lógica del robot está separada de planta_4_cintas.py.
- El modelo del Panda se carga desde MuJoCo Menagerie.
"""

import os
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np


# -----------------------------------------------------------------------------
# CONFIGURACIÓN INTERNA DEL PANDA
# -----------------------------------------------------------------------------
MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
PANDA_PREFIX = "panda_"
PANDA_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
PANDA_OPEN = 255.0
PANDA_CLOSED = 12.0

ROBOT_IK_DAMPING = 2.0e-3
ROBOT_MAX_QVEL = 1.15
ROBOT_POS_GAIN = 3.0
ROBOT_ORI_GAIN = 1.4
ROBOT_NULL_GAIN = 0.35
ROBOT_TARGET_TOL = 0.035
ROBOT_ORI_TOL_RAD = np.deg2rad(4.0)
ROBOT_STATE_DWELL = 0.28

# Límites específicos del modo IK cartesiano manual.
ROBOT_MANUAL_POS_GAIN = 3.2
ROBOT_MANUAL_ORI_GAIN = 1.8
ROBOT_MANUAL_MAX_QVEL = 1.25


def make_config(lane, branch_y, branch_speed):
    """Crea la configuración geométrica del robot sin acoplarla al módulo planta."""
    return {
        "base_x": -1.02,
        "base_y": 0.20,
        "base_z": 0.16,
        "base_yaw_deg": -90.0,
        "pick_object_ids": {0},
        "pick_trigger_y": 0.50,
        "pick_y": 0.38,
        "pick_x": -float(lane),
        "pick_approach_z": 0.58,
        "pick_z": 0.39,
        "place_x": -1.70,
        "place_y": float(branch_y),
        "place_approach_z": 0.58,
        "place_z": 0.39,
        "branch_speed": float(branch_speed),
    }


# -----------------------------------------------------------------------------
# CARGA E INTEGRACIÓN DEL MODELO MENAGERIE
# -----------------------------------------------------------------------------
def find_franka_model():
    """Localiza panda.xml en rutas habituales o mediante variable de entorno."""
    root = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get("MUJOCO_MENAGERIE_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        root / "vendor" / "mujoco_menagerie",
        root / "mujoco_menagerie",
        root.parent / "mujoco_menagerie",
        Path.home() / "mujoco_menagerie",
    ])

    for base in candidates:
        panda = base / "franka_emika_panda" / "panda.xml"
        if panda.exists():
            return panda.resolve()
    return None


def download_franka_model():
    """Descarga de forma sparse el modelo oficial si todavía no está local."""
    root = Path(__file__).resolve().parent
    vendor_root = root / "vendor"
    repo_dir = vendor_root / "mujoco_menagerie"
    panda_xml = repo_dir / "franka_emika_panda" / "panda.xml"

    if panda_xml.exists():
        return panda_xml.resolve()

    if shutil.which("git") is None:
        raise RuntimeError(
            "No se encontró Git. Instala git o define MUJOCO_MENAGERIE_PATH "
            "apuntando a un clon de google-deepmind/mujoco_menagerie."
        )

    vendor_root.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("[Franka] Descargando modelo oficial MuJoCo Menagerie...")
    try:
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none",
                "--sparse", MENAGERIE_REPO, str(repo_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", "franka_emika_panda"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        raise RuntimeError(
            "No se pudo descargar MuJoCo Menagerie. Puedes clonarlo manualmente con:\n"
            "  git clone https://github.com/google-deepmind/mujoco_menagerie.git\n"
            "y después definir MUJOCO_MENAGERIE_PATH=/ruta/al/mujoco_menagerie"
        ) from exc

    if not panda_xml.exists():
        raise RuntimeError("La descarga terminó, pero no aparece franka_emika_panda/panda.xml")
    return panda_xml.resolve()


def yaw_quaternion(degrees):
    """Quaternion MuJoCo w,x,y,z para una rotación yaw."""
    a = np.deg2rad(float(degrees)) * 0.5
    return [float(np.cos(a)), 0.0, 0.0, float(np.sin(a))]


def build_model_with_franka(plant_xml, cfg):
    """Compone la planta MJCF con el Panda oficial mediante MjSpec."""
    if plant_xml:
        plant_spec = mujoco.MjSpec.from_string(plant_xml)
    else:
        plant_path = Path(__file__).resolve().parent / "src" / "planta.xml"
        plant_spec = mujoco.MjSpec.from_file(str(plant_path))
    panda_path = find_franka_model()
    if panda_path is None:
        panda_path = download_franka_model()

    print(f"[Franka] Modelo: {panda_path}")
    panda_spec = mujoco.MjSpec.from_file(str(panda_path))

    panda_asset_dir = (panda_path.parent / "assets").resolve()
    for mesh in panda_spec.meshes:
        mesh_file = str(mesh.file) if mesh.file is not None else ""
        if mesh_file and not Path(mesh_file).is_absolute():
            mesh.file = str((panda_asset_dir / mesh_file).resolve())
    panda_spec.compiler.meshdir = ""

    # La planta ya tiene iluminación y la inicialización se hace proceduralmente.
    for light in list(panda_spec.lights):
        panda_spec.delete(light)
    for key in list(panda_spec.keys):
        panda_spec.delete(key)

    hand = panda_spec.body("hand")
    if hand is None:
        raise RuntimeError("El modelo Panda no contiene el body 'hand'.")
    hand.add_site(
        name="tcp",
        pos=[0.0, 0.0, 0.105],
        size=[0.012, 0.0, 0.0],
        rgba=[0.1, 0.9, 0.2, 0.8],
    )

    panda_spec.option.timestep = plant_spec.option.timestep
    panda_spec.option.integrator = plant_spec.option.integrator
    panda_spec.option.gravity[:] = plant_spec.option.gravity

    # Marcador visual del target cartesiano. Está unido al world y su posición
    # se actualiza en tiempo de ejecución desde set_ik_target_*().
    plant_spec.worldbody.add_site(
        name="ik_target_marker",
        pos=[0.0, 0.0, -2.0],
        size=[0.028, 0.0, 0.0],
        rgba=[1.0, 0.15, 0.10, 0.90],
    )

    mount = plant_spec.worldbody.add_frame(
        name="panda_mount",
        pos=[cfg["base_x"], cfg["base_y"], cfg["base_z"]],
        quat=yaw_quaternion(cfg["base_yaw_deg"]),
    )

    # copy_during_attach es write-only en versiones recientes de MuJoCo.
    try:
        plant_spec.copy_during_attach = True
    except (AttributeError, ValueError):
        try:
            plant_spec.copy_during_attaching = True
        except (AttributeError, ValueError):
            pass

    plant_spec.attach(panda_spec, frame=mount, prefix=PANDA_PREFIX)
    return plant_spec.compile()


# -----------------------------------------------------------------------------
# IDs, ESTADO E INICIALIZACIÓN
# -----------------------------------------------------------------------------
def get_robot_ids(model):
    """Resuelve IDs del Panda después del attach con prefijo panda_."""
    joint_ids = []
    qpos_addrs = []
    dof_addrs = []
    actuator_ids = []

    for j in range(1, 8):
        jname = f"{PANDA_PREFIX}joint{j}"
        aname = f"{PANDA_PREFIX}actuator{j}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if jid < 0 or aid < 0:
            raise RuntimeError(f"No se encontró {jname} / {aname} después de integrar el Panda.")
        joint_ids.append(jid)
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
        actuator_ids.append(aid)

    gripper_actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{PANDA_PREFIX}actuator8"
    )
    tcp_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{PANDA_PREFIX}tcp"
    )
    target_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "ik_target_marker"
    )
    if gripper_actuator_id < 0 or tcp_site_id < 0:
        raise RuntimeError("No se encontró el actuador del gripper o el site panda_tcp.")

    finger_qpos_addrs = []
    for fname in (f"{PANDA_PREFIX}finger_joint1", f"{PANDA_PREFIX}finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        if jid >= 0:
            finger_qpos_addrs.append(model.jnt_qposadr[jid])

    return {
        "joint_ids": np.asarray(joint_ids, dtype=int),
        "qpos_addrs": np.asarray(qpos_addrs, dtype=int),
        "dof_addrs": np.asarray(dof_addrs, dtype=int),
        "actuator_ids": np.asarray(actuator_ids, dtype=int),
        "gripper_actuator_id": int(gripper_actuator_id),
        "tcp_site_id": int(tcp_site_id),
        "target_site_id": int(target_site_id),
        "finger_qpos_addrs": finger_qpos_addrs,
    }


def new_robot_state():
    return {
        "state": "idle",
        "target_obj": -1,
        "grasped_obj": -1,
        "state_t0": 0.0,
        "grasp_offset": np.zeros(3, dtype=float),
        "home_xmat": None,
        "home_pos": None,
        "ik_target_pos": None,
        "ik_target_R": None,
        "ik_target_rpy_deg": None,
        "ik_orientation_frame": "world",
        "ik_enabled": False,
        "ik_reached": False,
        "last_ori_error_rad": 0.0,
        "last_ik_log_t": -1.0,
        "cycles": 0,
        "last_pos_error": 0.0,
    }


def initialize_robot(model, data, robot_ids, robot_state):
    """Inicializa el Panda en una postura estándar y abre el gripper."""
    for qadr, q in zip(robot_ids["qpos_addrs"], PANDA_HOME_Q):
        data.qpos[qadr] = q

    for dadr in robot_ids["dof_addrs"]:
        data.qvel[dadr] = 0.0

    for qadr in robot_ids["finger_qpos_addrs"]:
        data.qpos[qadr] = 0.04

    data.ctrl[robot_ids["actuator_ids"]] = PANDA_HOME_Q
    data.ctrl[robot_ids["gripper_actuator_id"]] = PANDA_OPEN
    mujoco.mj_forward(model, data)

    robot_state["home_xmat"] = data.site_xmat[robot_ids["tcp_site_id"]].reshape(3, 3).copy()
    robot_state["home_pos"] = data.site_xpos[robot_ids["tcp_site_id"]].copy()
    robot_state["state_t0"] = float(data.time)


# -----------------------------------------------------------------------------
# IK CARTESIANA GENERAL: POSICIÓN + ORIENTACIÓN
# -----------------------------------------------------------------------------
def rpy_deg_to_matrix(rpy_deg):
    """Convierte [roll, pitch, yaw] en grados a R=Rz(yaw)@Ry(pitch)@Rx(roll)."""
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=float).reshape(3))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def normalize_quaternion_wxyz(quat):
    q = np.asarray(quat, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("Quaternion IK inválido: norma cero.")
    return q / n


def quaternion_wxyz_to_matrix(quat):
    """Quaternion [w,x,y,z] -> matriz de rotación 3x3."""
    w, x, y, z = normalize_quaternion_wxyz(quat)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=float)


def set_ik_target_rpy(model, robot_ids, robot_state, position, rpy_deg, orientation_frame="world"):
    """
    Define un target 6D del TCP.

    position: [x,y,z] en WORLD, metros.
    rpy_deg: [roll,pitch,yaw] en grados.
    orientation_frame:
      - "world": RPY absoluto respecto a world.
      - "home":  RPY como incremento respecto a la orientación inicial del TCP.
    """
    pos = np.asarray(position, dtype=float).reshape(3)
    R_cmd = rpy_deg_to_matrix(rpy_deg)
    frame = str(orientation_frame).strip().lower()
    if frame == "home":
        if robot_state.get("home_xmat") is None:
            raise RuntimeError("El robot debe inicializarse antes de usar orientation_frame='home'.")
        R_target = robot_state["home_xmat"] @ R_cmd
    elif frame == "world":
        R_target = R_cmd
    else:
        raise ValueError("orientation_frame debe ser 'world' o 'home'.")

    robot_state["ik_target_pos"] = pos.copy()
    robot_state["ik_target_R"] = R_target.copy()
    robot_state["ik_target_rpy_deg"] = np.asarray(rpy_deg, dtype=float).reshape(3).copy()
    robot_state["ik_orientation_frame"] = frame
    robot_state["ik_enabled"] = True
    robot_state["ik_reached"] = False
    robot_state["state"] = "ik_manual"

    sid = robot_ids.get("target_site_id", -1)
    if sid is not None and sid >= 0:
        model.site_pos[sid] = pos


def set_ik_target_quat(model, robot_ids, robot_state, position, quat_wxyz):
    """Target 6D alternativo usando quaternion absoluto WORLD [w,x,y,z]."""
    pos = np.asarray(position, dtype=float).reshape(3)
    robot_state["ik_target_pos"] = pos.copy()
    robot_state["ik_target_R"] = quaternion_wxyz_to_matrix(quat_wxyz)
    robot_state["ik_target_rpy_deg"] = None
    robot_state["ik_orientation_frame"] = "world_quat"
    robot_state["ik_enabled"] = True
    robot_state["ik_reached"] = False
    robot_state["state"] = "ik_manual"
    sid = robot_ids.get("target_site_id", -1)
    if sid is not None and sid >= 0:
        model.site_pos[sid] = pos


def set_ik_target_home(model, robot_ids, robot_state):
    """Mueve el TCP de vuelta a la pose cartesiana registrada al inicializar."""
    if robot_state.get("home_pos") is None or robot_state.get("home_xmat") is None:
        raise RuntimeError("Pose HOME cartesiana no inicializada.")
    robot_state["ik_target_pos"] = robot_state["home_pos"].copy()
    robot_state["ik_target_R"] = robot_state["home_xmat"].copy()
    robot_state["ik_target_rpy_deg"] = np.zeros(3, dtype=float)
    robot_state["ik_orientation_frame"] = "home"
    robot_state["ik_enabled"] = True
    robot_state["ik_reached"] = False
    robot_state["state"] = "ik_manual"
    sid = robot_ids.get("target_site_id", -1)
    if sid is not None and sid >= 0:
        model.site_pos[sid] = robot_state["home_pos"]


def pose_errors(data, robot_ids, target_pos, target_R):
    site_id = robot_ids["tcp_site_id"]
    cur_pos = data.site_xpos[site_id].copy()
    cur_R = data.site_xmat[site_id].reshape(3, 3).copy()
    pos_err_vec = np.asarray(target_pos, dtype=float) - cur_pos
    ori_err_vec = rotation_error(cur_R, np.asarray(target_R, dtype=float))
    return pos_err_vec, ori_err_vec


def control_pose_ik(model, data, robot_ids, robot_state, target_pos, target_R,
                    pos_gain=ROBOT_MANUAL_POS_GAIN, ori_gain=ROBOT_MANUAL_ORI_GAIN,
                    max_qvel=ROBOT_MANUAL_MAX_QVEL):
    """Un paso de IK diferencial DLS 6D para posición y orientación del TCP."""
    site_id = robot_ids["tcp_site_id"]
    pos_err, ori_err = pose_errors(data, robot_ids, target_pos, target_R)

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    dofs = robot_ids["dof_addrs"]
    J = np.vstack((jacp[:, dofs], jacr[:, dofs]))

    twist = np.concatenate((float(pos_gain) * pos_err, float(ori_gain) * ori_err))
    A = J @ J.T + ROBOT_IK_DAMPING * np.eye(6)
    try:
        y = np.linalg.solve(A, twist)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(A, twist, rcond=None)[0]
        J_pinv = np.linalg.pinv(J)

    dq = J.T @ y
    q = data.qpos[robot_ids["qpos_addrs"]].copy()
    null = np.eye(7) - J_pinv @ J
    dq += null @ (ROBOT_NULL_GAIN * (PANDA_HOME_Q - q))
    dq = np.clip(dq, -float(max_qvel), float(max_qvel))

    q_cmd = q + dq * float(model.opt.timestep)
    for k, jid in enumerate(robot_ids["joint_ids"]):
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q_cmd[k] = np.clip(q_cmd[k], lo, hi)
    data.ctrl[robot_ids["actuator_ids"]] = q_cmd

    pos_norm = float(np.linalg.norm(pos_err))
    ori_norm = float(np.linalg.norm(ori_err))
    robot_state["last_pos_error"] = pos_norm
    robot_state["last_ori_error_rad"] = ori_norm
    robot_state["ik_reached"] = (pos_norm <= ROBOT_TARGET_TOL and ori_norm <= ROBOT_ORI_TOL_RAD)
    return pos_norm, ori_norm


def control_manual_ik(model, data, robot_ids, robot_state, gripper_open=True):
    """Controla continuamente el Panda hacia el target definido con set_ik_target_*()."""
    if not robot_state.get("ik_enabled", False):
        return False
    target_pos = robot_state.get("ik_target_pos")
    target_R = robot_state.get("ik_target_R")
    if target_pos is None or target_R is None:
        return False

    pos_err, ori_err = control_pose_ik(
        model, data, robot_ids, robot_state, target_pos, target_R
    )
    data.ctrl[robot_ids["gripper_actuator_id"]] = PANDA_OPEN if gripper_open else PANDA_CLOSED

    # Log limitado a 2 Hz para verificar convergencia sin saturar terminal.
    now = float(data.time)
    if robot_state["last_ik_log_t"] < 0.0 or now - robot_state["last_ik_log_t"] >= 0.5:
        print(
            f"[IK] TCP err_pos={pos_err*1000.0:6.1f} mm | "
            f"err_ori={np.rad2deg(ori_err):5.2f} deg | "
            f"{'TARGET OK' if robot_state['ik_reached'] else 'moviendo'}"
        )
        robot_state["last_ik_log_t"] = now
    return robot_state["ik_reached"]


def ik_target_text(robot_state):
    pos = robot_state.get("ik_target_pos")
    rpy = robot_state.get("ik_target_rpy_deg")
    if pos is None:
        return "IK sin target"
    ptxt = ", ".join(f"{v:.3f}" for v in pos)
    if rpy is None:
        return f"pos=[{ptxt}] orient=quaternion/world"
    rtxt = ", ".join(f"{v:.1f}" for v in rpy)
    return f"pos=[{ptxt}] rpy_deg=[{rtxt}] frame={robot_state.get('ik_orientation_frame','world')}"


# -----------------------------------------------------------------------------
# LÓGICA PICK-AND-PLACE
# -----------------------------------------------------------------------------
def robot_hold_force(pos, vel, cfg, max_force):
    """Detiene suavemente la pieza en la estación de recogida."""
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    fx = 18.0 * (cfg["pick_x"] - x) - 5.0 * vx
    fy = 18.0 * (cfg["pick_y"] - y) - 5.0 * vy
    return float(np.clip(fx, -max_force, max_force)), float(np.clip(fy, -max_force, max_force))


def watch_for_cube(data, body_ids, segments, robot_state, cfg):
    """Reserva una pieza objetivo al entrar en la zona de recogida."""
    if robot_state["state"] != "idle" or robot_state["target_obj"] >= 0:
        return

    for i in sorted(cfg["pick_object_ids"]):
        if segments[i] != 3:
            continue
        pos = data.xpos[body_ids[i]]
        if cfg["pick_y"] - 0.04 <= float(pos[1]) <= cfg["pick_trigger_y"]:
            robot_state["target_obj"] = i
            robot_state["state"] = "approach_pick"
            robot_state["state_t0"] = float(data.time)
            print(f"[Panda] Pieza obj{i} reservada en Cinta 4.")
            return


def state_target_position(state_name, cfg):
    """Target cartesiano del TCP para cada fase del pick-and-place."""
    targets = {
        "approach_pick": np.array([cfg["pick_x"], cfg["pick_y"], cfg["pick_approach_z"]]),
        "descend_pick": np.array([cfg["pick_x"], cfg["pick_y"], cfg["pick_z"]]),
        "close": np.array([cfg["pick_x"], cfg["pick_y"], cfg["pick_z"]]),
        "lift": np.array([cfg["pick_x"], cfg["pick_y"], cfg["pick_approach_z"]]),
        "transfer": np.array([cfg["place_x"], cfg["place_y"], cfg["place_approach_z"]]),
        "descend_place": np.array([cfg["place_x"], cfg["place_y"], cfg["place_z"]]),
        "open": np.array([cfg["place_x"], cfg["place_y"], cfg["place_z"]]),
        "retract": np.array([cfg["place_x"], cfg["place_y"], cfg["place_approach_z"]]),
    }
    return targets.get(state_name)


def set_robot_state(robot_state, name, sim_time):
    robot_state["state"] = name
    robot_state["state_t0"] = float(sim_time)


def tcp_position(data, robot_ids):
    return data.site_xpos[robot_ids["tcp_site_id"]].copy()


def robot_target_reached(data, robot_ids, robot_state, cfg):
    target = state_target_position(robot_state["state"], cfg)
    if target is None:
        return False
    err = float(np.linalg.norm(target - tcp_position(data, robot_ids)))
    robot_state["last_pos_error"] = err
    return err <= ROBOT_TARGET_TOL


def grasp_object(model, data, robot_ids, robot_state):
    i = robot_state["target_obj"]
    if i < 0:
        return
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
    robot_state["grasped_obj"] = i
    robot_state["grasp_offset"] = data.xpos[bid].copy() - tcp_position(data, robot_ids)
    print(f"[Panda] obj{i} agarrado.")


def follow_tcp_with_grasped_object(model, data, robot_ids, robot_state):
    """Grasp-assist: sigue el TCP mientras el gripper porta la pieza."""
    i = robot_state["grasped_obj"]
    if i < 0:
        return

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]
    pos = tcp_position(data, robot_ids) + robot_state["grasp_offset"]
    data.qpos[qadr:qadr+3] = pos
    data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr:dadr+6] = 0.0


def release_object_to_branch(model, data, robot_state, segments, object_rest_z, cfg):
    """Deposita la pieza centrada en Cinta 5 y cede el transporte a la banda."""
    i = robot_state["grasped_obj"]
    if i < 0:
        return

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]
    data.qpos[qadr:qadr+3] = [cfg["place_x"], cfg["place_y"], object_rest_z[i]]
    data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr:dadr+6] = [-cfg["branch_speed"], 0.0, 0.0, 0.0, 0.0, 0.0]
    segments[i] = 4
    robot_state["grasped_obj"] = -1
    print(f"[Panda] obj{i} depositado en Cinta 5 -> horno.")


def rotation_error(current, desired):
    """Error rotacional 3D a partir de dos matrices de rotación."""
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def control_robot_ik(model, data, robot_ids, robot_state, cfg):
    """IK 6D del pick-and-place automático reutilizando el solver cartesiano general."""
    state = robot_state["state"]
    act_ids = robot_ids["actuator_ids"]

    if state == "idle":
        data.ctrl[act_ids] = PANDA_HOME_Q
        data.ctrl[robot_ids["gripper_actuator_id"]] = PANDA_OPEN
        return

    target = state_target_position(state, cfg)
    if target is None:
        return

    desired_R = robot_state["home_xmat"]
    control_pose_ik(
        model, data, robot_ids, robot_state, target, desired_R,
        pos_gain=ROBOT_POS_GAIN, ori_gain=ROBOT_ORI_GAIN, max_qvel=ROBOT_MAX_QVEL,
    )

    if state in ("close", "lift", "transfer", "descend_place"):
        data.ctrl[robot_ids["gripper_actuator_id"]] = PANDA_CLOSED
    else:
        data.ctrl[robot_ids["gripper_actuator_id"]] = PANDA_OPEN


def update_robot_state_machine(model, data, body_ids, robot_ids, robot_state, segments, object_rest_z, cfg):
    """Secuencia procedural: aproximar -> coger -> levantar -> depositar -> retirar."""
    state = robot_state["state"]
    now = float(data.time)

    if state == "idle":
        watch_for_cube(data, body_ids, segments, robot_state, cfg)
        return

    elapsed = now - robot_state["state_t0"]

    # Recuperación si una fase de IK no converge.
    if elapsed > 7.0:
        i = robot_state["grasped_obj"] if robot_state["grasped_obj"] >= 0 else robot_state["target_obj"]
        if i >= 0:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            qadr = model.jnt_qposadr[jid]
            dadr = model.jnt_dofadr[jid]
            data.qpos[qadr:qadr+3] = [cfg["pick_x"], cfg["pick_y"], object_rest_z[i]]
            data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[dadr:dadr+6] = 0.0
            segments[i] = 3
        print(f"[Panda] Timeout en fase {state}; reinicio seguro del ciclo.")
        robot_state["target_obj"] = -1
        robot_state["grasped_obj"] = -1
        set_robot_state(robot_state, "idle", now)
        return

    reached = robot_target_reached(data, robot_ids, robot_state, cfg)

    if state == "approach_pick" and reached:
        set_robot_state(robot_state, "descend_pick", now)
    elif state == "descend_pick" and reached:
        set_robot_state(robot_state, "close", now)
    elif state == "close" and elapsed >= ROBOT_STATE_DWELL:
        grasp_object(model, data, robot_ids, robot_state)
        set_robot_state(robot_state, "lift", now)
    elif state == "lift" and reached:
        set_robot_state(robot_state, "transfer", now)
    elif state == "transfer" and reached:
        set_robot_state(robot_state, "descend_place", now)
    elif state == "descend_place" and reached:
        release_object_to_branch(model, data, robot_state, segments, object_rest_z, cfg)
        set_robot_state(robot_state, "open", now)
    elif state == "open" and elapsed >= ROBOT_STATE_DWELL:
        set_robot_state(robot_state, "retract", now)
    elif state == "retract" and reached:
        robot_state["cycles"] += 1
        robot_state["target_obj"] = -1
        set_robot_state(robot_state, "idle", now)
        print(f"[Panda] Ciclo completado. Total={robot_state['cycles']}")
