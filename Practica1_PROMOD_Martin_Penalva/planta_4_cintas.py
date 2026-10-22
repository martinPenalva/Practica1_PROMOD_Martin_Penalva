#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planta industrial con 4 cintas transportadoras formando un circuito cuadrado.
- MuJoCo + Python
- Programación procedural: NO usa clases / POO
- 4 cintas: inferior, derecha, superior, izquierda
- Objetos transferidos automáticamente entre cintas
- Orientación de los objetos bloqueada: se trasladan pero no rotan
- Los cilindros se mantienen exactamente sobre la línea central de cada banda
- 4 cámaras de inspección, una por cinta, más una para la rama del horno
- Isla robotizada junto a la mitad de la cinta 4
- Franka Emika Panda oficial de MuJoCo Menagerie sobre la isla
- Control del Panda separado en robot.py (procedural, sin POO)
- IK cartesiana general del Panda: target XYZ + orientación RPY o quaternion
- Modo de prueba IK manual para mover el efector final a una pose dada
- El pick-and-place automático puede reactivarse deshabilitando el modo de prueba IK
- Cinta lateral de salida hacia un horno
- Mosaico OpenCV con las cámaras de la planta
- Ventana 3D de MuJoCo opcional

Controles:
    Q / ESC : salir
    P       : pausar/reanudar toda la simulación
    1       : arrancar/detener Cinta 1
    2       : arrancar/detener Cinta 2
    3       : arrancar/detener Cinta 3
    4       : arrancar/detener Cinta 4
    5       : arrancar/detener Cinta 5 (horno)
    R       : reiniciar objetos
    S       : guardar captura del mosaico
    T       : enviar de nuevo la pose IK de prueba
    H       : mover el TCP a la pose cartesiana HOME
"""

import time
from pathlib import Path

import cv2
import numpy as np
import mujoco
import mujoco.viewer

import robot


# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
SHOW_MUJOCO_VIEWER = True
SHOW_CAMERA_MOSAIC = True

CAM_W = 360
CAM_H = 260
CAM_FPS = 20.0

BELT_SPEED = 0.55          # m/s, velocidad objetivo de los objetos

# -----------------------------------------------------------------------------
# BANDERAS DE MARCHA / PARO DE CADA CINTA
# -----------------------------------------------------------------------------
# True  -> cinta en marcha
# False -> cinta detenida
# Estas variables pueden modificarse directamente antes de ejecutar el programa
# y también con las teclas 1..5 durante la simulación.
BANDERA_CINTA_1 = True      # cinta inferior (+X)
BANDERA_CINTA_2 = True      # cinta derecha (+Y)
BANDERA_CINTA_3 = True      # cinta superior (-X)
BANDERA_CINTA_4 = True      # cinta izquierda (-Y)
BANDERA_CINTA_5 = True      # cinta hacia el horno (-X)

VEL_KP = 7.0               # control de velocidad
CENTER_KP = 12.0           # mantiene el objeto centrado en la cinta
TURN_KP = 16.0             # ayuda a realizar el giro en las esquinas
MAX_FORCE = 12.0           # N por eje

BELT_HALF_LENGTH = 1.45
BELT_HALF_WIDTH = 0.28
BELT_Z = 0.12
BELT_TOP_Z = 0.18
LANE = 1.45                 # coordenada de la línea central de cada cinta
TURN_START = 1.08           # comienza a orientar el objeto hacia la siguiente cinta
CORNER_SWITCH = 1.27        # cambio efectivo de cinta dentro de la placa de esquina
RAIL_END_GAP = 0.06       # separación uniforme antes de cada zona de transferencia
RAIL_HALF_LENGTH = LANE - BELT_HALF_WIDTH - RAIL_END_GAP
RAIL_THICKNESS = 0.035
RAIL_HALF_HEIGHT = 0.10
RAIL_CENTER_Z = 0.22

# Cámaras de inspección. El poste queda fuera de la cinta, pero el cabezal
# óptico queda sobre el borde de la banda y mira casi perpendicularmente hacia
# la superficie. El FOV estrecho evita ver gran parte del suelo/planta.
CAM_HEIGHT = 0.92
CAM_FOVY = 32.0
CAM_LENS_SIDE_OFFSET = 0.10
CAM_POLE_SIDE_OFFSET = BELT_HALF_WIDTH + 0.45

# Cámara específica de la cinta del horno: el cabezal queda completamente
# FUERA del ancho de la banda y mira hacia su línea central.
CAM_OVEN_HEIGHT = 0.78
CAM_OVEN_SIDE_OFFSET = 0.46
CAM_OVEN_FOVY = 34.0

# Cámara 4 (izquierda): se coloca completamente FUERA de la banda y además
# desplazada hacia la mitad superior de la Cinta 4. De esta forma no invade
# ni la banda ni la intersección Cinta 4 -> Cinta 5/Panda.
CAM_LEFT_INSPECTION_Y = 0.82
CAM_LEFT_HEAD_CLEARANCE = 0.20
CAM_LEFT_POLE_CLEARANCE = 0.52

# Evita que las piezas roten por contacto, rozamiento o durante los giros.
# Conservamos freejoint para que sigan teniendo dinámica XYZ, pero forzamos
# quaternion identidad y velocidad angular nula después de cada paso físico.
LOCK_OBJECT_ROTATION = True

# Objetos cilíndricos del modelo (azul y verde). En tramos rectos se fija
# exactamente su coordenada transversal a la línea central de la banda.
CYLINDER_OBJECT_IDS = {1, 3}
LOCK_CYLINDERS_TO_CENTERLINE = True

# Dimensiones de las piezas. Los cilindros azul (obj1) y verde (obj3)
# conservan su radio pero tienen ahora la MITAD de la altura anterior.
OBJECT_HALF = 0.085
OBJ1_RADIUS = 0.075
OBJ1_HALF_HEIGHT = 0.045      # antes 0.090 -> altura total 0.09 m
OBJ2_HALF_Z = 0.070
OBJ3_RADIUS = 0.085
OBJ3_HALF_HEIGHT = 0.0375     # antes 0.075 -> altura total 0.075 m
OBJ4_HALF_Z = 0.065
OBJECT_CLEARANCE_Z = 0.01
OBJECT_REST_Z = [
    BELT_TOP_Z + OBJECT_HALF + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ1_HALF_HEIGHT + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ2_HALF_Z + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ3_HALF_HEIGHT + OBJECT_CLEARANCE_Z,
    BELT_TOP_Z + OBJ4_HALF_Z + OBJECT_CLEARANCE_Z,
]

# Rama lateral desde la Cinta 4 hacia el horno.
# La boca se deja completamente libre: el robot deposita la pieza sobre la
# Cinta 5 y la banda se encarga del resto. Ya no se hace un desvío dinámico
# desde la Cinta 4, que era la causa principal del atasco observado.
BRANCH_SPEED = 0.48
BRANCH_Y = 0.0
BRANCH_HALF_LENGTH = 1.15
BRANCH_HALF_WIDTH = 0.32
BRANCH_CENTER_X = -(LANE + BRANCH_HALF_LENGTH - 0.05)
BRANCH_OPEN_HALF = 0.38
BRANCH_RAIL_HALF = (RAIL_HALF_LENGTH - BRANCH_OPEN_HALF) / 2.0
BRANCH_RAIL_CENTER_Y = (RAIL_HALF_LENGTH + BRANCH_OPEN_HALF) / 2.0
BRANCH_RAIL_START_X = -2.00
BRANCH_RAIL_END_X = -3.48
BRANCH_RAIL_CENTER_X = 0.5 * (BRANCH_RAIL_START_X + BRANCH_RAIL_END_X)
BRANCH_RAIL_HALF_LENGTH = 0.5 * abs(BRANCH_RAIL_END_X - BRANCH_RAIL_START_X)

# Apertura de la barandilla interior de Cinta 4 para que el Panda pueda
# alcanzar el cubo sin atravesar una protección física.
PICK_RAIL_OPEN_Y_MIN = -0.10
PICK_RAIL_OPEN_Y_MAX = 0.72
LEFT_RAIL_UPPER_CENTER_Y = 0.5 * (RAIL_HALF_LENGTH + PICK_RAIL_OPEN_Y_MAX)
LEFT_RAIL_UPPER_HALF = 0.5 * (RAIL_HALF_LENGTH - PICK_RAIL_OPEN_Y_MAX)
LEFT_RAIL_LOWER_CENTER_Y = 0.5 * (-RAIL_HALF_LENGTH + PICK_RAIL_OPEN_Y_MIN)
LEFT_RAIL_LOWER_HALF = 0.5 * (PICK_RAIL_OPEN_Y_MIN + RAIL_HALF_LENGTH)

# Configuración geométrica del Panda. La lógica completa del brazo está en
# robot.py y sigue siendo 100 % procedural (sin clases / sin POO).
ROBOT_CONFIG = robot.make_config(
    lane=LANE,
    branch_y=BRANCH_Y,
    branch_speed=BRANCH_SPEED,
)

# -----------------------------------------------------------------------------
# PRUEBA DE IK CARTESIANA DEL PANDA
# -----------------------------------------------------------------------------
# True: desactiva temporalmente el pick-and-place y controla el TCP con una
# pose objetivo 6D. False: recupera el comportamiento automático de la planta.
ROBOT_IK_TEST_MODE = True

# Posición del TCP en coordenadas WORLD [x, y, z] (metros).
ROBOT_IK_TARGET_POSITION = np.array([-1.45, 0.05, 0.62], dtype=float)

# Orientación [roll, pitch, yaw] en grados.
# Con frame="home", estos ángulos son un incremento respecto a la orientación
# inicial del efector. Esto hace la primera prueba mucho más segura.
# Para una orientación absoluta respecto a WORLD, cambia el frame a "world".
ROBOT_IK_TARGET_RPY_DEG = np.array([0.0, 0.0, 20.0], dtype=float)
ROBOT_IK_ORIENTATION_FRAME = "home"   # "home" o "world"
ROBOT_IK_GRIPPER_OPEN = True

# Isla robotizada: la plataforma toma la pose base directamente de robot.py
# para que no haya dos configuraciones distintas que puedan desalinearse.
ISLAND_X = ROBOT_CONFIG["base_x"]
ISLAND_Y = ROBOT_CONFIG["base_y"]
ISLAND_Z = ROBOT_CONFIG["base_z"]
ISLAND_HALF_X = 0.18
ISLAND_HALF_Y = 0.38

# Horno al final de la nueva cinta.
OVEN_CENTER_X = -4.05
OVEN_ENTRY_X = -3.58

OUTPUT_DIR = Path(__file__).resolve().parent / "capturas"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CintaTransportadora:
    """Encapsula el estado y configuración de una cinta."""

    def __init__(self, nombre, velocidad, direccion=(1.0, 0.0), activa=True):
        self.nombre = nombre
        self.velocidad = float(velocidad)
        self.direccion = tuple(direccion)
        self.activa = bool(activa)

    def arrancar(self):
        self.activa = True

    def detener(self):
        self.activa = False

    def conmutar(self):
        self.activa = not self.activa
        return self.activa

    def esta_activa(self):
        return self.activa


class ObjetoIndustrial:
    """Representa una pieza que circula por la planta."""

    def __init__(self, identificador, forma, segmento=0, posicion=(0.0, 0.0, 0.0), cuerpo_id=None):
        self.identificador = identificador
        self.forma = forma
        self.segmento = int(segmento)
        self.posicion = list(posicion)
        self.cuerpo_id = cuerpo_id

    def cambiar_segmento(self, nuevo_segmento):
        self.segmento = int(nuevo_segmento)

    def __repr__(self):
        return f"ObjetoIndustrial({self.identificador}, forma={self.forma}, segmento={self.segmento})"


class CamaraInspeccion:
    """Camera wrapper with detection state for its assigned belt."""

    def __init__(self, nombre, cinta):
        self.nombre = nombre
        self.cinta = cinta
        self.objeto_detectado = False
        self.forma_detectada = None

    def actualizar(self, imagen):
        self.objeto_detectado = False
        self.forma_detectada = None
        if imagen is not None:
            self.objeto_detectado = imagen.size > 0
            self.forma_detectada = "circulo" if imagen.size > 0 else None
        if self.objeto_detectado:
            self.cinta.detener()
        return imagen


class Horno:
    """Encapsula la parte final de la línea y la temperatura."""

    def __init__(self, nombre="Horno", temperatura_inicial=20.0, temperatura_min=15.0, temperatura_max=250.0):
        self.nombre = nombre
        self.numero_piezas = 0
        self.estado = "inactivo"
        self.temperatura = float(temperatura_inicial)
        self.temperatura_minima = float(temperatura_min)
        self.temperatura_maxima = float(temperatura_max)
        self.geom_id = None

    def registrar_pieza(self):
        self.numero_piezas += 1
        self.estado = "ocupado"

    def aumentar_temperatura(self, incremento):
        self.temperatura = min(self.temperatura + float(incremento), self.temperatura_maxima)
        return self.temperatura

    def disminuir_temperatura(self, decremento):
        self.temperatura = max(self.temperatura - float(decremento), self.temperatura_minima)
        return self.temperatura

    def obtener_temperatura(self):
        return self.temperatura


class PlantaIndustrial:
    """Agrupa cintas, cámaras, objetos y horno."""

    def __init__(self):
        self.cintas = {
            1: CintaTransportadora("Cinta 1", BELT_SPEED, (1.0, 0.0), True),
            2: CintaTransportadora("Cinta 2", BELT_SPEED, (0.0, 1.0), True),
            3: CintaTransportadora("Cinta 3", BELT_SPEED, (-1.0, 0.0), True),
            4: CintaTransportadora("Cinta 4", BELT_SPEED, (0.0, -1.0), True),
            5: CintaTransportadora("Cinta 5", BRANCH_SPEED, (-1.0, 0.0), True),
        }
        self.camaras = {
            1: CamaraInspeccion("cam_bottom", self.cintas[1]),
            2: CamaraInspeccion("cam_right", self.cintas[2]),
            3: CamaraInspeccion("cam_top", self.cintas[3]),
            4: CamaraInspeccion("cam_left", self.cintas[4]),
            5: CamaraInspeccion("cam_oven", self.cintas[5]),
        }
        self.objetos = [
            ObjetoIndustrial("obj0", "cubo", 3, (-LANE, 0.95, OBJECT_REST_Z[0])),
            ObjetoIndustrial("obj1", "cilindro", 1, (LANE, -0.72, OBJECT_REST_Z[1])),
            ObjetoIndustrial("obj2", "caja", 2, (0.58, LANE, OBJECT_REST_Z[2])),
            ObjetoIndustrial("obj3", "cilindro", 0, (-0.62, -LANE, OBJECT_REST_Z[3])),
            ObjetoIndustrial("obj4", "cubo", 2, (-0.45, LANE, OBJECT_REST_Z[4])),
        ]
        self.horno = Horno("horno_principal")
        self.segments = [3, 1, 2, 3, 0]
        self.oven_counts = [0, 0, 0, 0, 0]

    def estado_cintas(self):
        return [self.cintas[i].esta_activa() for i in range(1, 6)]

    def conmutar_cinta(self, numero):
        if numero not in self.cintas:
            raise ValueError(f"Número de cinta inválido: {numero}")
        return self.cintas[numero].conmutar()

    def actualizar(self):
        for cinta in self.cintas.values():
            if not cinta.esta_activa():
                continue
        for camara in self.camaras.values():
            camara.actualizar(None)

    def run(self):
        return run()


PLANTA = PlantaIndustrial()


# -----------------------------------------------------------------------------
# GEOMETRÍA XML DE LA PLANTA
# -----------------------------------------------------------------------------
def camera_xyaxes(cam_pos, target):
    """Devuelve los ejes X/Y de una cámara MuJoCo que mira hacia `target`."""
    cam = np.asarray(cam_pos, dtype=float)
    tgt = np.asarray(target, dtype=float)

    # La cámara de MuJoCo mira a lo largo de su eje local -Z.
    z_axis = cam - tgt
    z_axis /= np.linalg.norm(z_axis)

    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    vals = np.concatenate([x_axis, y_axis])
    return " ".join(f"{v:.6f}" for v in vals)


def build_mjcf():
    """Construye el MJCF como texto. Todo se genera proceduralmente."""

    # Cámaras de inspección: el soporte/poste está a un lado de cada cinta,
    # mientras que el objetivo óptico queda cerca del borde de la banda. Así
    # la imagen contiene esencialmente banda + pieza y resulta adecuada para
    # reconocimiento posterior de color y forma.
    cam_bottom = (0.0, -(LANE + CAM_LENS_SIDE_OFFSET), CAM_HEIGHT)
    cam_right  = (LANE + CAM_LENS_SIDE_OFFSET, 0.0, CAM_HEIGHT)
    cam_top    = (0.0, LANE + CAM_LENS_SIDE_OFFSET, CAM_HEIGHT)

    # La cámara 4 NO queda encima de la banda. Su x está más allá del borde
    # exterior de la Cinta 4 y su y se aleja de la intersección con Cinta 5.
    cam_left = (
        -(LANE + BELT_HALF_WIDTH + CAM_LEFT_HEAD_CLEARANCE),
        CAM_LEFT_INSPECTION_Y,
        CAM_HEIGHT,
    )
    cam_oven   = (BRANCH_CENTER_X, BRANCH_Y - CAM_OVEN_SIDE_OFFSET, CAM_OVEN_HEIGHT)

    # Se apunta al centro de una zona local de inspección de la banda. El
    # pequeño offset hace que la cámara siga estando físicamente a un lado,
    # pero la dirección de visión sea casi cenital.
    target_bottom = (0.0, -LANE, BELT_TOP_Z + 0.015)
    target_right  = (LANE, 0.0, BELT_TOP_Z + 0.015)
    target_top    = (0.0, LANE, BELT_TOP_Z + 0.015)
    target_left   = (-LANE, CAM_LEFT_INSPECTION_Y, BELT_TOP_Z + 0.015)
    target_oven   = (BRANCH_CENTER_X, BRANCH_Y, BELT_TOP_Z + 0.015)

    pole_bottom_y = -(LANE + CAM_POLE_SIDE_OFFSET)
    pole_right_x  =  (LANE + CAM_POLE_SIDE_OFFSET)
    pole_top_y    =  (LANE + CAM_POLE_SIDE_OFFSET)
    pole_left_x   = -(LANE + BELT_HALF_WIDTH + CAM_LEFT_POLE_CLEARANCE)
    pole_left_y   = CAM_LEFT_INSPECTION_Y
    pole_oven_y   = BRANCH_Y - (BRANCH_HALF_WIDTH + 0.52)

    xy_bottom = camera_xyaxes(cam_bottom, target_bottom)
    xy_right  = camera_xyaxes(cam_right, target_right)
    xy_top    = camera_xyaxes(cam_top, target_top)
    xy_left   = camera_xyaxes(cam_left, target_left)
    xy_oven   = camera_xyaxes(cam_oven, target_oven)

# -----------------------------------------------------------------------------
# UTILIDADES DE OBJETOS
# -----------------------------------------------------------------------------
def get_ids(model):
    body_ids = []
    dof_addrs = []
    for i in range(5):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj{i}")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
        body_ids.append(bid)
        dof_addrs.append(model.jnt_dofadr[jid])
    return body_ids, dof_addrs



def reset_objects(model, data, segments):
    """Reinicia las 5 piezas. El cubo rojo comienza cerca de la isla."""
    starts = [
        (-LANE, 0.95, OBJECT_REST_Z[0], 3),   # cubo rojo: llega pronto a la estación Panda
        ( LANE, -0.72, OBJECT_REST_Z[1], 1),
        ( 0.58,  LANE, OBJECT_REST_Z[2], 2),
        (-0.62, -LANE, OBJECT_REST_Z[3], 0),  # cilindro verde lejos de la estación robot
        (-0.45,  LANE, OBJECT_REST_Z[4], 2),
    ]

    for i, (x, y, z, seg) in enumerate(starts):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid]

        data.qpos[qadr:qadr+3] = [x, y, z]
        data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[dadr:dadr+6] = 0.0
        segments[i] = seg

    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)



def lock_object_rotations(model, data):
    """Mantiene todas las piezas con orientación fija respecto al mundo."""
    if not LOCK_OBJECT_ROTATION:
        return

    for i in range(5):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid]
        data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[dadr+3:dadr+6] = 0.0


def lock_cylinders_to_belt_center(model, data, segments):
    """Mantiene los cilindros en la línea central de los tramos rectos."""
    if not LOCK_CYLINDERS_TO_CENTERLINE:
        return

    for i in CYLINDER_OBJECT_IDS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid]
        x = float(data.qpos[qadr])
        y = float(data.qpos[qadr + 1])
        seg = segments[i]

        if seg == 0 and x < TURN_START:
            data.qpos[qadr + 1] = -LANE
            data.qvel[dadr + 1] = 0.0
        elif seg == 1 and y < TURN_START:
            data.qpos[qadr] = LANE
            data.qvel[dadr] = 0.0
        elif seg == 2 and x > -TURN_START:
            data.qpos[qadr + 1] = LANE
            data.qvel[dadr + 1] = 0.0
        elif seg == 3 and y > -TURN_START:
            data.qpos[qadr] = -LANE
            data.qvel[dadr] = 0.0
        elif seg == 4:
            data.qpos[qadr + 1] = BRANCH_Y
            data.qvel[dadr + 1] = 0.0


def update_segment(pos, seg, obj_index):
    """Máquina de estados del circuito. La Cinta 5 solo la activa el robot."""
    x, y = float(pos[0]), float(pos[1])

    if seg == 0 and x >= CORNER_SWITCH:
        return 1
    if seg == 1 and y >= CORNER_SWITCH:
        return 2
    if seg == 2 and x <= -CORNER_SWITCH:
        return 3
    if seg == 3 and y <= -CORNER_SWITCH:
        return 0
    return seg


def cinta_activa(seg):
    """Devuelve la bandera de marcha de la cinta asociada al segmento 0..4."""
    if seg == 0:
        return BANDERA_CINTA_1
    if seg == 1:
        return BANDERA_CINTA_2
    if seg == 2:
        return BANDERA_CINTA_3
    if seg == 3:
        return BANDERA_CINTA_4
    if seg == 4:
        return BANDERA_CINTA_5
    return False


def estado_cintas():
    """Estado actual de las cinco banderas, en orden Cinta 1..5."""
    return [
        BANDERA_CINTA_1,
        BANDERA_CINTA_2,
        BANDERA_CINTA_3,
        BANDERA_CINTA_4,
        BANDERA_CINTA_5,
    ]


def conmutar_cinta(numero):
    """Conmuta en tiempo de ejecución una bandera de marcha/parada."""
    global BANDERA_CINTA_1, BANDERA_CINTA_2, BANDERA_CINTA_3
    global BANDERA_CINTA_4, BANDERA_CINTA_5

    if numero == 1:
        BANDERA_CINTA_1 = not BANDERA_CINTA_1
        return BANDERA_CINTA_1
    if numero == 2:
        BANDERA_CINTA_2 = not BANDERA_CINTA_2
        return BANDERA_CINTA_2
    if numero == 3:
        BANDERA_CINTA_3 = not BANDERA_CINTA_3
        return BANDERA_CINTA_3
    if numero == 4:
        BANDERA_CINTA_4 = not BANDERA_CINTA_4
        return BANDERA_CINTA_4
    if numero == 5:
        BANDERA_CINTA_5 = not BANDERA_CINTA_5
        return BANDERA_CINTA_5
    raise ValueError(f"Número de cinta inválido: {numero}")


def transport_force(pos, vel, seg, cinta_encendida=None):
    """Fuerza de transporte de las 4 cintas y de la rama al horno."""
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])

    fx = 0.0
    fy = 0.0

    # Si la bandera de esta cinta está a False, se aplica frenado viscoso
    # sobre el plano XY. La física sigue activa y el resto de cintas/robot
    # continúan funcionando de forma independiente.
    if cinta_encendida is None:
        cinta_encendida = cinta_activa(seg)
    if not cinta_encendida:
        fx = VEL_KP * (0.0 - vx)
        fy = VEL_KP * (0.0 - vy)
        return (
            float(np.clip(fx, -MAX_FORCE, MAX_FORCE)),
            float(np.clip(fy, -MAX_FORCE, MAX_FORCE)),
        )

    if seg == 0:  # inferior: +X -> +Y
        alpha = np.clip((x - TURN_START) / max(CORNER_SWITCH - TURN_START, 1e-6), 0.0, 1.0)
        desired_vx = BELT_SPEED * (1.0 - 0.45 * alpha)
        desired_vy = BELT_SPEED * (0.85 * alpha)
        fx = VEL_KP * (desired_vx - vx)
        fy = VEL_KP * (desired_vy - vy) + (1.0 - alpha) * CENTER_KP * (-LANE - y)

    elif seg == 1:  # derecha: +Y -> -X
        alpha = np.clip((y - TURN_START) / max(CORNER_SWITCH - TURN_START, 1e-6), 0.0, 1.0)
        desired_vx = -BELT_SPEED * (0.85 * alpha)
        desired_vy = BELT_SPEED * (1.0 - 0.45 * alpha)
        fx = VEL_KP * (desired_vx - vx) + (1.0 - alpha) * CENTER_KP * (LANE - x)
        fy = VEL_KP * (desired_vy - vy)

    elif seg == 2:  # superior: -X -> -Y
        alpha = np.clip((-x - TURN_START) / max(CORNER_SWITCH - TURN_START, 1e-6), 0.0, 1.0)
        desired_vx = -BELT_SPEED * (1.0 - 0.45 * alpha)
        desired_vy = -BELT_SPEED * (0.85 * alpha)
        fx = VEL_KP * (desired_vx - vx)
        fy = VEL_KP * (desired_vy - vy) + (1.0 - alpha) * CENTER_KP * (LANE - y)

    elif seg == 3:  # izquierda: -Y -> +X en esquina inferior
        alpha = np.clip((-y - TURN_START) / max(CORNER_SWITCH - TURN_START, 1e-6), 0.0, 1.0)
        desired_vx = BELT_SPEED * (0.85 * alpha)
        desired_vy = -BELT_SPEED * (1.0 - 0.45 * alpha)
        fx = VEL_KP * (desired_vx - vx) + (1.0 - alpha) * CENTER_KP * (-LANE - x)
        fy = VEL_KP * (desired_vy - vy)

    elif seg == 4:  # Cinta 5 al horno: -X
        desired_vx = -BRANCH_SPEED
        desired_vy = 0.0
        fx = VEL_KP * (desired_vx - vx)
        fy = VEL_KP * (desired_vy - vy) + CENTER_KP * (BRANCH_Y - y)

    return float(np.clip(fx, -MAX_FORCE, MAX_FORCE)), float(np.clip(fy, -MAX_FORCE, MAX_FORCE))



def send_object_back_to_line(model, data, i, dadr, segments):
    """Al entrar en el horno, reaparece en Cinta 1 para mantener flujo continuo."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
    qadr = model.jnt_qposadr[jid]
    x0 = -1.05 + 0.18 * (i % 3)
    data.qpos[qadr:qadr+3] = [x0, -LANE, OBJECT_REST_Z[i]]
    data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr:dadr+6] = 0.0
    segments[i] = 0


def apply_transport(model, data, body_ids, dof_addrs, segments, oven_counts, robot_state):
    """Aplica transporte; el cubo reservado se inmoviliza hasta que lo toma el Panda."""
    data.xfrc_applied[:] = 0.0

    for i, bid in enumerate(body_ids):
        pos = data.xpos[bid]
        dadr = dof_addrs[i]
        vel = data.qvel[dadr:dadr+3]

        # Si el robot ya porta la pieza, la cinta no debe aplicar ninguna fuerza.
        if i == robot_state["grasped_obj"]:
            continue

        # Cubo reservado: mantenerlo centrado y detenido en la zona de pick.
        if i == robot_state["target_obj"] and robot_state["state"] != "idle":
            fx, fy = robot.robot_hold_force(pos, vel, ROBOT_CONFIG, MAX_FORCE)
        else:
            segments[i] = update_segment(pos, segments[i], i)
            fx, fy = transport_force(pos, vel, segments[i])

        data.xfrc_applied[bid, 0] = fx
        data.xfrc_applied[bid, 1] = fy

        if segments[i] == 4 and pos[0] <= OVEN_ENTRY_X:
            oven_counts[i] += 1
            send_object_back_to_line(model, data, i, dadr, segments)
            continue

        if pos[2] < -0.1 or abs(pos[0]) > 5.0 or abs(pos[1]) > 2.4:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr:qadr+3] = [-0.8 + 0.25*i, -LANE, OBJECT_REST_Z[i]]
            data.qpos[qadr+3:qadr+7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[dadr:dadr+6] = 0.0
            segments[i] = 0


# -----------------------------------------------------------------------------
# VISUALIZACIÓN DE LAS 4 CÁMARAS
# -----------------------------------------------------------------------------
def add_label(image, text, status_text=None):
    img = image.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(img, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 245, 245), 2, cv2.LINE_AA)
    if status_text:
        cv2.putText(img, status_text, (12, img.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return img


def render_camera_mosaic(renderer, data, paused, sim_time, oven_total, robot_state):
    cams = [
        ("cam_bottom", "CINTA 1 · INFERIOR · +X"),
        ("cam_right",  "CINTA 2 · DERECHA · +Y"),
        ("cam_top",    "CINTA 3 · SUPERIOR · -X"),
        ("cam_left",   "CINTA 4 · IZQUIERDA · -Y"),
        ("cam_oven",   "CINTA 5 · HORNO · -X"),
        ("overview",   "VISTA GENERAL"),
    ]

    frames = []
    flags = estado_cintas()
    flags_txt = " ".join(f"C{i+1}:{'ON' if v else 'OFF'}" for i, v in enumerate(flags))
    status = (f"t={sim_time:6.1f}s · {'PAUSA' if paused else 'EN MARCHA'} · "
              f"{flags_txt} · Panda={robot_state['state']} · "
              f"ciclos={robot_state['cycles']} · horno={oven_total}")

    for cam_name, title in cams:
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(add_label(bgr, title, status))

    top = np.hstack((frames[0], frames[1], frames[2]))
    bottom = np.hstack((frames[3], frames[4], frames[5]))
    mosaic = np.vstack((top, bottom))

    cv2.putText(mosaic, "Q/ESC salir | P pausa | 1-5 cintas ON/OFF | R reset | S captura",
                (12, mosaic.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return mosaic


# -----------------------------------------------------------------------------
# ACTUADORES DE LOS RODILLOS DE LA PLANTA
# -----------------------------------------------------------------------------
def initialize_belt_rollers(model, data):
    """Actualiza los rodillos según la bandera independiente de cada cinta."""
    f1, f2, f3, f4, f5 = estado_cintas()
    roller_commands = {
        "belt_act_b1":  5.5 if f1 else 0.0,
        "belt_act_b2":  5.5 if f1 else 0.0,
        "belt_act_r1": -5.5 if f2 else 0.0,
        "belt_act_r2": -5.5 if f2 else 0.0,
        "belt_act_t1": -5.5 if f3 else 0.0,
        "belt_act_t2": -5.5 if f3 else 0.0,
        "belt_act_l1":  5.5 if f4 else 0.0,
        "belt_act_l2":  5.5 if f4 else 0.0,
        "belt_act_o1": -5.5 if f5 else 0.0,
        "belt_act_o2": -5.5 if f5 else 0.0,
    }
    for name, value in roller_commands.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"No se encontró el actuador de rodillo '{name}'.")
        data.ctrl[aid] = value


# -----------------------------------------------------------------------------
# BUCLE PRINCIPAL
# -----------------------------------------------------------------------------
def run():
    model = robot.build_model_with_franka(build_mjcf(), ROBOT_CONFIG)
    data = mujoco.MjData(model)

    overview_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

    body_ids, dof_addrs = get_ids(model)
    robot_ids = robot.get_robot_ids(model)
    robot_state = robot.new_robot_state()
    segments = [3, 1, 2, 3, 0]
    oven_counts = [0, 0, 0, 0, 0]

    reset_objects(model, data, segments)
    robot.initialize_robot(model, data, robot_ids, robot_state)

    if ROBOT_IK_TEST_MODE:
        robot.set_ik_target_rpy(
            model, robot_ids, robot_state,
            ROBOT_IK_TARGET_POSITION,
            ROBOT_IK_TARGET_RPY_DEG,
            orientation_frame=ROBOT_IK_ORIENTATION_FRAME,
        )
        mujoco.mj_forward(model, data)

    initialize_belt_rollers(model, data)

    renderer = None
    if SHOW_CAMERA_MOSAIC:
        renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
        cv2.namedWindow("Camaras de la planta", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camaras de la planta", CAM_W*3, CAM_H*2)

    paused = False
    running = True
    last_cam_time = -1.0

    print("\n=== PLANTA ROBOTIZADA: 4 CINTAS + PANDA + HORNO ===")
    print("Circuito: inferior -> derecha -> superior -> izquierda -> inferior")
    print("Estacion robot: Cinta 4 -> Franka Panda -> Cinta 5 -> horno")
    print(f"Piezas que recoge el Panda: {sorted(ROBOT_CONFIG['pick_object_ids'])}")
    print("El desvio dinamico anterior se ha eliminado para evitar el atasco en la interseccion.")
    print("Q/ESC: salir | P: pausa global | 1..5: ON/OFF cintas | R: reset | S: captura")
    if ROBOT_IK_TEST_MODE:
        print("Modo IK TEST: T=pose objetivo configurada | H=pose HOME cartesiana")
        print("Target IK:", robot.ik_target_text(robot_state))
    else:
        print("Modo robot: pick-and-place automático")
    print("Banderas iniciales:", " ".join(
        f"C{i+1}={'ON' if v else 'OFF'}" for i, v in enumerate(estado_cintas())
    ))
    print(f"Camaras: FOV={CAM_FOVY:.0f} deg, altura={CAM_HEIGHT:.2f} m")
    print(f"Bloqueo de rotacion de objetos: {LOCK_OBJECT_ROTATION}")
    print(f"Cilindros centrados: {LOCK_CYLINDERS_TO_CENTERLINE}")
    print(f"Panda base=({ROBOT_CONFIG['base_x']:.2f}, {ROBOT_CONFIG['base_y']:.2f}, {ROBOT_CONFIG['base_z']:.2f}) yaw={ROBOT_CONFIG['base_yaw_deg']:.0f} deg")
    print("Programacion procedural: no se usan clases.\n")

    viewer_ctx = None
    if SHOW_MUJOCO_VIEWER:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        viewer_ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer_ctx.cam.fixedcamid = overview_id

    try:
        while running:
            if viewer_ctx is not None and not viewer_ctx.is_running():
                break

            step_start = time.perf_counter()

            if not paused:
                if ROBOT_IK_TEST_MODE:
                    # Modo de validación: se ignora temporalmente la máquina de estados
                    # pick-and-place y se manda directamente una pose 6D al TCP.
                    apply_transport(
                        model, data, body_ids, dof_addrs, segments, oven_counts, robot_state
                    )
                    robot.control_manual_ik(
                        model, data, robot_ids, robot_state,
                        gripper_open=ROBOT_IK_GRIPPER_OPEN,
                    )
                else:
                    # Modo automático original: Cinta 4 -> Panda -> Cinta 5.
                    robot.update_robot_state_machine(
                        model, data, body_ids, robot_ids, robot_state, segments,
                        OBJECT_REST_Z, ROBOT_CONFIG
                    )
                    apply_transport(
                        model, data, body_ids, dof_addrs, segments, oven_counts, robot_state
                    )
                    robot.control_robot_ik(model, data, robot_ids, robot_state, ROBOT_CONFIG)
                    robot.follow_tcp_with_grasped_object(model, data, robot_ids, robot_state)

                # Física.
                mujoco.mj_step(model, data)

                # Estabilización de piezas.
                lock_object_rotations(model, data)
                lock_cylinders_to_belt_center(model, data, segments)
                if not ROBOT_IK_TEST_MODE:
                    robot.follow_tcp_with_grasped_object(model, data, robot_ids, robot_state)
                mujoco.mj_forward(model, data)

            if viewer_ctx is not None:
                viewer_ctx.sync()

            if SHOW_CAMERA_MOSAIC and renderer is not None:
                if last_cam_time < 0.0 or (data.time - last_cam_time) >= (1.0 / CAM_FPS) or paused:
                    mosaic = render_camera_mosaic(
                        renderer, data, paused, data.time, sum(oven_counts), robot_state
                    )
                    cv2.imshow("Camaras de la planta", mosaic)
                    last_cam_time = data.time

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    running = False
                elif key == ord('p'):
                    paused = not paused
                    print("PAUSA GLOBAL" if paused else "SIMULACIÓN REANUDADA")
                elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                    numero_cinta = key - ord('0')
                    activa = conmutar_cinta(numero_cinta)
                    initialize_belt_rollers(model, data)
                    print(f"Cinta {numero_cinta}: {'ON' if activa else 'OFF'}")
                elif key == ord('r'):
                    reset_objects(model, data, segments)
                    oven_counts[:] = [0, 0, 0, 0, 0]
                    robot_state.clear()
                    robot_state.update(robot.new_robot_state())
                    robot.initialize_robot(model, data, robot_ids, robot_state)
                    if ROBOT_IK_TEST_MODE:
                        robot.set_ik_target_rpy(
                            model, robot_ids, robot_state,
                            ROBOT_IK_TARGET_POSITION, ROBOT_IK_TARGET_RPY_DEG,
                            orientation_frame=ROBOT_IK_ORIENTATION_FRAME,
                        )
                        mujoco.mj_forward(model, data)
                    initialize_belt_rollers(model, data)
                    print("Planta y Panda reiniciados")
                elif key == ord('t') and ROBOT_IK_TEST_MODE:
                    robot.set_ik_target_rpy(
                        model, robot_ids, robot_state,
                        ROBOT_IK_TARGET_POSITION, ROBOT_IK_TARGET_RPY_DEG,
                        orientation_frame=ROBOT_IK_ORIENTATION_FRAME,
                    )
                    mujoco.mj_forward(model, data)
                    print("[IK] Target configurado:", robot.ik_target_text(robot_state))
                elif key == ord('h') and ROBOT_IK_TEST_MODE:
                    robot.set_ik_target_home(model, robot_ids, robot_state)
                    mujoco.mj_forward(model, data)
                    print("[IK] Target HOME cartesiano activado")
                elif key == ord('s'):
                    if 'mosaic' in locals():
                        filename = OUTPUT_DIR / f"mosaico_{int(time.time())}.png"
                        cv2.imwrite(str(filename), mosaic)
                        print(f"Captura guardada: {filename}")

            elapsed = time.perf_counter() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        if renderer is not None:
            renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
