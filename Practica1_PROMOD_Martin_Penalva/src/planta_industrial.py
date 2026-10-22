"""Coordina la planta MuJoCo y sus componentes orientados a objetos."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

import planta_4_cintas as simulacion
import robot as controlador_panda
from .camara_inspeccion import CamaraInspeccion
from .cinta_transportadora import CintaTransportadora
from .horno import Horno
from .objeto_industrial import Caja, Cilindro, ObjetoIndustrial
from .RobotPanda import RobotPanda


class PlantaIndustrial:
    """Posee el modelo MuJoCo y coordina cintas, visión, horno y robot."""

    NOMBRES_CAMARAS = {
        1: "cam_bottom",
        2: "cam_right",
        3: "cam_top",
        4: "cam_left",
        5: "cam_oven",
    }
    FORMAS = ("cubo", "cilindro", "caja", "cilindro", "cubo")

    def __init__(self, mostrar_visor: bool = True, mostrar_camaras: bool = True) -> None:
        self.mostrar_visor = mostrar_visor
        self.mostrar_camaras = mostrar_camaras
        self.cintas = {
            numero: CintaTransportadora(
                f"Cinta {numero}",
                simulacion.BELT_SPEED if numero < 5 else simulacion.BRANCH_SPEED,
                direccion,
            )
            for numero, direccion in enumerate(
                ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)),
                start=1,
            )
        }
        self.camaras = {
            numero: CamaraInspeccion(nombre, self.cintas[numero])
            for numero, nombre in self.NOMBRES_CAMARAS.items()
        }
        tipos = (Caja, Cilindro, Caja, Cilindro, Caja)
        self.objetos: list[ObjetoIndustrial] = [
            clase(f"obj{indice}", segmento=segmento, posicion=(0.0, 0.0, 0.0))
            for indice, (clase, segmento) in enumerate(
                zip(tipos, (3, 1, 2, 3, 0))
            )
        ]
        for pieza, segmento in zip(self.objetos, (3, 1, 2, 3, 0)):
            pieza.forma = self.FORMAS[int(pieza.id[3:])]
            pieza.tipo = pieza.forma
            pieza.segmento = segmento

        self.horno = Horno("horno_principal")
        self.robot = RobotPanda()
        self.model = None
        self.data = None
        self.renderer = None
        self.visor = None
        self.robot_ids = None
        self.robot_state = None
        self.body_ids = None
        self.dof_addrs = None
        self.segments = [3, 1, 2, 3, 0]
        self.oven_counts = [0, 0, 0, 0, 0]
        self.paused = False
        self.running = False
        self.sim_time = 0.0
        self._manual_stops: set[int] = set()
        self._camera_latched = {numero: False for numero in self.camaras}
        self._inspection_deadlines = {numero: 0.0 for numero in self.camaras}
        self._last_camera_time = -1.0
        self._last_camera_time = -1.0
        self._mosaic = None
        self._output_dir = Path(__file__).resolve().parent.parent / "capturas"

    def inicializar(self) -> None:
        """Construye el modelo, crea MuJoCo data y prepara el Panda."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.visor is not None:
            self.visor.close()
            self.visor = None
        if self.model is None:
            config_robot = simulacion.ROBOT_CONFIG
            self.model = controlador_panda.build_model_with_franka(
                None, config_robot
            )
        self.data = mujoco.MjData(self.model)
        self.body_ids, self.dof_addrs = simulacion.get_ids(self.model)
        self.robot_ids = controlador_panda.get_robot_ids(self.model)
        self.robot_state = controlador_panda.new_robot_state()
        self.segments = [3, 1, 2, 3, 0]
        self.oven_counts = [0, 0, 0, 0, 0]
        simulacion.reset_objects(self.model, self.data, self.segments)
        self.robot.inicializar(
            self.model, self.data, self.robot_ids, self.robot_state
        )

        if simulacion.ROBOT_IK_TEST_MODE:
            controlador_panda.set_ik_target_rpy(
                self.model,
                self.robot_ids,
                self.robot_state,
                simulacion.ROBOT_IK_TARGET_POSITION,
                simulacion.ROBOT_IK_TARGET_RPY_DEG,
                orientation_frame=simulacion.ROBOT_IK_ORIENTATION_FRAME,
            )
            mujoco.mj_forward(self.model, self.data)

        oven_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "oven_hot_geom"
        )
        if oven_geom_id < 0:
            raise RuntimeError("No se encontró la geometría visual oven_hot_geom del horno.")
        self.horno = Horno("horno_principal")
        self.horno.geom_id = int(oven_geom_id)
        self.horno.actualizar_color_mujoco(self.model)

        for cinta in self.cintas.values():
            cinta.iniciar()
        self._manual_stops.clear()
        self._camera_latched = {numero: False for numero in self.camaras}
        self._inspection_deadlines = {numero: 0.0 for numero in self.camaras}
        self._actualizar_objetos()
        self._actualizar_rodillos()

        if self.mostrar_camaras:
            self.renderer = mujoco.Renderer(
                self.model, height=simulacion.CAM_H, width=simulacion.CAM_W
            )
            cv2.namedWindow("Camaras de la planta", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "Camaras de la planta",
                simulacion.CAM_W * 3,
                simulacion.CAM_H * 2,
            )
        self.sim_time = 0.0
        self.paused = False
        self.running = True

    def _actualizar_objetos(self) -> None:
        if self.model is None or self.data is None:
            return
        for cinta in self.cintas.values():
            cinta.objetos.clear()
        for indice, pieza in enumerate(self.objetos):
            body_id = self.body_ids[indice]
            pieza.cuerpo_id = int(body_id)
            pieza.segmento = int(self.segments[indice])
            pieza.posicion = self.data.xpos[body_id].tolist()
            if pieza.segmento in self.cintas:
                self.cintas[pieza.segmento + 1].agregar_objeto(pieza)

    def estado_cintas(self) -> list[bool]:
        return [self.cintas[numero].esta_activa() for numero in sorted(self.cintas)]

    def conmutar_cinta(self, numero: int) -> bool:
        if numero not in self.cintas:
            raise ValueError(f"Número de cinta inválido: {numero}")
        if self.cintas[numero].esta_activa():
            self._manual_stops.add(numero)
            self.cintas[numero].detener()
        else:
            self._manual_stops.discard(numero)
            if not self._camera_latched[numero]:
                self.cintas[numero].iniciar()
        self._actualizar_rodillos()
        return self.cintas[numero].esta_activa()

    def _actualizar_rodillos(self) -> None:
        if self.model is None or self.data is None:
            return
        comandos = {
            "belt_act_b1": 5.5 if self.cintas[1].esta_activa() else 0.0,
            "belt_act_b2": 5.5 if self.cintas[1].esta_activa() else 0.0,
            "belt_act_r1": -5.5 if self.cintas[2].esta_activa() else 0.0,
            "belt_act_r2": -5.5 if self.cintas[2].esta_activa() else 0.0,
            "belt_act_t1": -5.5 if self.cintas[3].esta_activa() else 0.0,
            "belt_act_t2": -5.5 if self.cintas[3].esta_activa() else 0.0,
            "belt_act_l1": 5.5 if self.cintas[4].esta_activa() else 0.0,
            "belt_act_l2": 5.5 if self.cintas[4].esta_activa() else 0.0,
            "belt_act_o1": -5.5 if self.cintas[5].esta_activa() else 0.0,
            "belt_act_o2": -5.5 if self.cintas[5].esta_activa() else 0.0,
        }
        for nombre, valor in comandos.items():
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nombre
            )
            if actuator_id < 0:
                raise RuntimeError(f"No se encontró el actuador de rodillo {nombre!r}.")
            self.data.ctrl[actuator_id] = valor

    def _actualizar_transporte(self) -> None:
        self.data.xfrc_applied[:] = 0.0
        for indice, body_id in enumerate(self.body_ids):
            posicion = self.data.xpos[body_id]
            dof = self.dof_addrs[indice]
            velocidad = self.data.qvel[dof:dof + 3]

            if indice == self.robot_state["grasped_obj"]:
                continue
            if indice == self.robot_state["target_obj"] and self.robot_state["state"] != "idle":
                fx, fy = controlador_panda.robot_hold_force(
                    posicion, velocidad, simulacion.ROBOT_CONFIG, simulacion.MAX_FORCE
                )
            else:
                self.segments[indice] = simulacion.update_segment(
                    posicion, self.segments[indice], indice
                )
                cinta = self.cintas[self.segments[indice] + 1]
                fx, fy = simulacion.transport_force(
                    posicion,
                    velocidad,
                    self.segments[indice],
                    cinta_encendida=cinta.esta_activa(),
                )
            self.data.xfrc_applied[body_id, 0] = fx
            self.data.xfrc_applied[body_id, 1] = fy

            if (
                self.segments[indice] == 4
                and posicion[0] <= simulacion.OVEN_ENTRY_X
            ):
                self.oven_counts[indice] += 1
                pieza = self.objetos[indice]
                self.horno.registrar_pieza(pieza)
                simulacion.send_object_back_to_line(
                    self.model, self.data, indice, dof, self.segments
                )

            if (
                posicion[2] < -0.1
                or abs(posicion[0]) > 5.0
                or abs(posicion[1]) > 2.4
            ):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"obj{indice}_free",
                )
                qpos = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos:qpos + 3] = [
                    -0.8 + 0.25 * indice,
                    -simulacion.LANE,
                    simulacion.OBJECT_REST_Z[indice],
                ]
                self.data.qpos[qpos + 3:qpos + 7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[dof:dof + 6] = 0.0
                self.segments[indice] = 0

    def _actualizar_camaras(self) -> list[np.ndarray]:
        imagenes = []
        for numero, camara in self.camaras.items():
            imagen = camara.actualizar(
                self.renderer, self.data, detener_cinta=False
            )
            imagenes.append(imagen)
            if camara.objeto_detectado:
                if not self._camera_latched[numero]:
                    self.cintas[numero].detener()
                    self._camera_latched[numero] = True
                    self._inspection_deadlines[numero] = self.sim_time + 0.35
                elif (
                    self.sim_time >= self._inspection_deadlines[numero]
                    and numero not in self._manual_stops
                ):
                    self.cintas[numero].iniciar()
            else:
                self._camera_latched[numero] = False
                if numero not in self._manual_stops:
                    self.cintas[numero].iniciar()
        self._actualizar_rodillos()
        return imagenes

    def _crear_mosaico(self, imagenes: list[np.ndarray]) -> np.ndarray:
        paneles = []
        for (numero, camara), imagen in zip(self.camaras.items(), imagenes):
            panel = imagen.copy()
            estado = "PIEZA DETECTADA" if camara.objeto_detectado else "SIN PIEZA"
            cv2.putText(
                panel,
                f"CINTA {numero} - {camara.forma_detectada or estado}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            paneles.append(panel)
        self.renderer.update_scene(self.data, camera="overview")
        overview = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        paneles.append(overview)
        superior = np.hstack(paneles[:3])
        inferior = np.hstack(paneles[3:])
        return np.vstack((superior, inferior))

    def actualizar(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Llama a inicializar() antes de actualizar la planta.")
        if self.paused:
            return

        for indice, cinta in self.cintas.items():
            if indice in self._manual_stops:
                cinta.detener()
        if simulacion.ROBOT_IK_TEST_MODE:
            self._actualizar_transporte()
            self.robot.controlar_manual(
                gripper_open=simulacion.ROBOT_IK_GRIPPER_OPEN
            )
        else:
            self.robot.actualizar_ciclo_automatico(
                self.body_ids,
                self.segments,
                simulacion.OBJECT_REST_Z,
                simulacion.ROBOT_CONFIG,
            )
            self._actualizar_transporte()

        mujoco.mj_step(self.model, self.data)
        simulacion.lock_object_rotations(self.model, self.data)
        simulacion.lock_cylinders_to_belt_center(
            self.model, self.data, self.segments
        )
        if not simulacion.ROBOT_IK_TEST_MODE:
            self.robot.seguir_pieza()
        self.sim_time = float(self.data.time)
        self.horno.actualizar(incremento=0.01, model=self.model)
        mujoco.mj_forward(self.model, self.data)
        self._actualizar_objetos()
        self.robot.sincronizar_pose(simulacion.ROBOT_CONFIG)

    def procesar_teclado(self, tecla: int) -> None:
        if tecla in (ord("q"), 27):
            self.running = False
        elif tecla == ord("p"):
            self.paused = not self.paused
        elif tecla == ord("u"):
            temperatura = self.horno.aumentar_temperatura(40.0)
            self.horno.actualizar_color_mujoco(self.model)
            print(
                f"Temperatura del horno: {temperatura:.1f} °C "
                f"({self.horno.color})"
            )
        elif tecla == ord("d"):
            temperatura = self.horno.disminuir_temperatura(40.0)
            self.horno.actualizar_color_mujoco(self.model)
            print(
                f"Temperatura del horno: {temperatura:.1f} °C "
                f"({self.horno.color})"
            )
        elif tecla in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            numero = tecla - ord("0")
            self.conmutar_cinta(numero)
        elif tecla == ord("r"):
            self.segments = [3, 1, 2, 3, 0]
            self.oven_counts = [0, 0, 0, 0, 0]
            simulacion.reset_objects(self.model, self.data, self.segments)
            self.robot_state.clear()
            self.robot_state.update(controlador_panda.new_robot_state())
            self.robot.inicializar(
                self.model, self.data, self.robot_ids, self.robot_state
            )
            if simulacion.ROBOT_IK_TEST_MODE:
                controlador_panda.set_ik_target_rpy(
                    self.model,
                    self.robot_ids,
                    self.robot_state,
                    simulacion.ROBOT_IK_TARGET_POSITION,
                    simulacion.ROBOT_IK_TARGET_RPY_DEG,
                    orientation_frame=simulacion.ROBOT_IK_ORIENTATION_FRAME,
                )
                mujoco.mj_forward(self.model, self.data)
            self.horno = Horno("horno_principal")
            self.horno.geom_id = int(
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "oven_hot_geom"
                )
            )
            self.horno.actualizar_color_mujoco(self.model)
            self._manual_stops.clear()
            self._camera_latched = {numero: False for numero in self.camaras}
            self._inspection_deadlines = {numero: 0.0 for numero in self.camaras}
            for cinta in self.cintas.values():
                cinta.iniciar()
            self._actualizar_rodillos()
            self._actualizar_objetos()
        elif tecla == ord("s") and self._mosaic is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            destino = self._output_dir / f"mosaico_{int(time.time())}.png"
            if not cv2.imwrite(str(destino), self._mosaic):
                raise OSError(f"No se pudo guardar la captura en {destino}.")
            print(f"Captura guardada: {destino}")
        elif tecla == ord("t") and simulacion.ROBOT_IK_TEST_MODE:
            controlador_panda.set_ik_target_rpy(
                self.model,
                self.robot_ids,
                self.robot_state,
                simulacion.ROBOT_IK_TARGET_POSITION,
                simulacion.ROBOT_IK_TARGET_RPY_DEG,
                orientation_frame=simulacion.ROBOT_IK_ORIENTATION_FRAME,
            )
        elif tecla == ord("h") and simulacion.ROBOT_IK_TEST_MODE:
            controlador_panda.set_ik_target_home(
                self.model, self.robot_ids, self.robot_state
            )
        if tecla in (ord("t"), ord("h")):
            self.robot.sincronizar_pose(simulacion.ROBOT_CONFIG)

    def ejecutar(self, max_frames: int | None = None) -> None:
        self.inicializar()
        overview_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
        )
        if self.mostrar_visor:
            self.visor = mujoco.viewer.launch_passive(self.model, self.data)
            self.visor.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.visor.cam.fixedcamid = overview_id
        frames = 0
        try:
            while self.running and (max_frames is None or frames < max_frames):
                if self.visor is not None and not self.visor.is_running():
                    break
                inicio = time.perf_counter()
                self.actualizar()
                if self.visor is not None:
                    self.visor.sync()
                if self.renderer is not None:
                    if (
                        self._last_camera_time < 0
                        or self.sim_time - self._last_camera_time >= 1.0 / simulacion.CAM_FPS
                        or self.paused
                    ):
                        imagenes = self._actualizar_camaras()
                        self._mosaic = self._crear_mosaico(imagenes)
                        cv2.imshow("Camaras de la planta", self._mosaic)
                        self._last_camera_time = self.sim_time
                    self.procesar_teclado(cv2.waitKey(1) & 0xFF)
                tiempo_restante = self.model.opt.timestep - (time.perf_counter() - inicio)
                if tiempo_restante > 0:
                    time.sleep(tiempo_restante)
                frames += 1
        finally:
            self.cerrar()

    def cerrar(self) -> None:
        if self.visor is not None:
            self.visor.close()
            self.visor = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        cv2.destroyAllWindows()
