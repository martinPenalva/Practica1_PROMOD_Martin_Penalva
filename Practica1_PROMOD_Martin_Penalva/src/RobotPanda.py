"""Control lógico del robot Panda en la simulación orientada a objetos."""

from __future__ import annotations

import robot as controlador_panda


class RobotPanda:
    """Adapta el controlador IK proporcionado para su uso desde la planta."""

    def __init__(self, velocidad: float = 1.0) -> None:
        if velocidad <= 0:
            raise ValueError("La velocidad del robot debe ser mayor que cero.")
        self.pose_actual = [0.0, 0.0, 0.0]
        self.pose_objetivo = [0.0, 0.0, 0.0]
        self.ik_activa = True
        self.velocidad = float(velocidad)
        self.pieza_actual = None
        self.model = None
        self.data = None
        self.robot_ids = None
        self.robot_state = None

    def inicializar(self, model, data, robot_ids, robot_state=None) -> None:
        self.model = model
        self.data = data
        self.robot_ids = robot_ids
        self.robot_state = robot_state or controlador_panda.new_robot_state()
        controlador_panda.initialize_robot(
            self.model, self.data, self.robot_ids, self.robot_state
        )

    def controlar_manual(self, gripper_open: bool = True) -> bool:
        self._validar_vinculo()
        return controlador_panda.control_manual_ik(
            self.model,
            self.data,
            self.robot_ids,
            self.robot_state,
            gripper_open=gripper_open,
        )

    def actualizar_ciclo_automatico(
        self, body_ids, segments, object_rest_z, config
    ) -> None:
        self._validar_vinculo()
        controlador_panda.update_robot_state_machine(
            self.model,
            self.data,
            body_ids,
            self.robot_ids,
            self.robot_state,
            segments,
            object_rest_z,
            config,
        )
        controlador_panda.control_robot_ik(
            self.model, self.data, self.robot_ids, self.robot_state, config
        )
        controlador_panda.follow_tcp_with_grasped_object(
            self.model, self.data, self.robot_ids, self.robot_state
        )

    def seguir_pieza(self) -> None:
        self._validar_vinculo()
        controlador_panda.follow_tcp_with_grasped_object(
            self.model, self.data, self.robot_ids, self.robot_state
        )

    def sincronizar_pose(self, config=None) -> None:
        self._validar_vinculo()
        tcp_id = self.robot_ids["tcp_site_id"]
        self.pose_actual = self.data.site_xpos[tcp_id].tolist()
        objetivo = self.robot_state.get("ik_target_pos")
        if objetivo is None and config is not None:
            objetivo = controlador_panda.state_target_position(
                self.robot_state["state"], config
            )
        if objetivo is not None:
            self.pose_objetivo = list(objetivo)

    def _validar_vinculo(self) -> None:
        if any(
            componente is None
            for componente in (
                self.model,
                self.data,
                self.robot_ids,
                self.robot_state,
            )
        ):
            raise RuntimeError("El robot Panda no está vinculado a un modelo MuJoCo.")

    def mover_a(self, pose: tuple[float, float, float] | list[float]) -> bool:
        if len(pose) != 3:
            raise ValueError("La pose objetivo debe contener x, y y z.")
        self.pose_objetivo = [float(coordenada) for coordenada in pose]
        return True

    def coger_pieza(self, pieza) -> bool:
        if self.pieza_actual is not None:
            return False
        self.mover_a(pieza.posicion)
        self.pose_actual = self.pose_objetivo.copy()
        self.pieza_actual = pieza
        pieza.segmento = -1
        return True

    def colocar_pieza(self, destino) -> bool:
        if self.pieza_actual is None:
            return False

        if hasattr(destino, "registrar_pieza"):
            destino.registrar_pieza(self.pieza_actual)
        elif hasattr(destino, "agregar_objeto"):
            destino.agregar_objeto(self.pieza_actual)
        elif hasattr(destino, "posicion"):
            self.pieza_actual.posicion = list(destino.posicion)
        else:
            raise TypeError("El destino debe ser un horno o un objeto con posición.")

        self.pose_objetivo = list(self.pieza_actual.posicion)
        self.pose_actual = self.pose_objetivo.copy()
        self.pieza_actual = None
        return True

    def actualizar(self, dt: float = 0.1) -> None:
        if dt < 0:
            raise ValueError("El intervalo de actualización no puede ser negativo.")
        if not self.ik_activa or dt == 0:
            return

        desplazamiento = [
            objetivo - actual
            for actual, objetivo in zip(self.pose_actual, self.pose_objetivo)
        ]
        distancia = sum(componente * componente for componente in desplazamiento) ** 0.5
        paso = self.velocidad * dt
        if distancia <= paso or distancia == 0:
            self.pose_actual = self.pose_objetivo.copy()
            return
        escala = paso / distancia
        self.pose_actual = [
            actual + componente * escala
            for actual, componente in zip(self.pose_actual, desplazamiento)
        ]
