"""Modela el horno industrial y su cola de piezas."""

from __future__ import annotations


class Horno:
    """Calienta y procesa las piezas que recibe."""

    def __init__(
        self,
        nombre: str = "Horno",
        temperatura_inicial: float = 20.0,
        temperatura_min: float = 15.0,
        temperatura_max: float = 250.0,
    ) -> None:
        self.nombre = nombre
        self.temperatura = float(temperatura_inicial)
        self.temperatura_minima = float(temperatura_min)
        self.temperatura_maxima = float(temperatura_max)
        if self.temperatura_minima > self.temperatura_maxima:
            raise ValueError("La temperatura mínima no puede superar la máxima.")
        self.piezas = 0
        self.piezas_procesadas = 0
        self.color = "azul"
        self.activo = False
        self.geom_id = None
        self._cola: list[object | None] = []

    @property
    def numero_piezas(self) -> int:
        return self.piezas

    @numero_piezas.setter
    def numero_piezas(self, valor: int) -> None:
        self.piezas = int(valor)

    @property
    def estado(self) -> str:
        if self._cola:
            return "ocupado"
        return "activo" if self.activo else "inactivo"

    @estado.setter
    def estado(self, valor: str) -> None:
        self.activo = valor not in {"inactivo", "apagado", "vacío", "vacio"}

    def registrar_pieza(self, pieza=None) -> None:
        self._cola.append(pieza)
        self.piezas += 1
        self.activo = True
        self.actualizar_color()

    def aumentar_temperatura(self, incremento: float = 5.0) -> float:
        if incremento < 0:
            raise ValueError("El incremento de temperatura no puede ser negativo.")
        self.temperatura = min(self.temperatura + float(incremento), self.temperatura_maxima)
        self.actualizar_color()
        return self.temperatura

    def disminuir_temperatura(self, decremento: float = 5.0) -> float:
        if decremento < 0:
            raise ValueError("El decremento de temperatura no puede ser negativo.")
        self.temperatura = max(self.temperatura - float(decremento), self.temperatura_minima)
        self.actualizar_color()
        return self.temperatura

    def actualizar_color(self, model=None) -> str:
        if self.temperatura < 80:
            self.color = "azul"
        elif self.temperatura < 180:
            self.color = "naranja"
        else:
            self.color = "rojo"
        if model is not None:
            self.actualizar_color_mujoco(model)
        return self.color

    def actualizar_color_mujoco(self, model) -> None:
        if self.geom_id is None:
            return
        colores = {
            "azul": (0.1, 0.3, 1.0, 1.0),
            "naranja": (1.0, 0.6, 0.1, 1.0),
            "rojo": (1.0, 0.1, 0.1, 1.0),
        }
        color = self.actualizar_color()
        model.geom_rgba[self.geom_id] = colores[color]

    def actualizar(self, incremento: float = 5.0, model=None) -> None:
        self.activo = bool(self._cola)
        if self._cola:
            self.aumentar_temperatura(incremento)
            if self.temperatura >= 180:
                self.procesar_pieza()
        else:
            self.disminuir_temperatura(incremento)
        if model is not None:
            self.actualizar_color_mujoco(model)

    def procesar_pieza(self, pieza=None):
        if self.temperatura < 180:
            return None
        if pieza is None:
            if not self._cola:
                return None
            pieza = self._cola.pop(0)
        elif pieza in self._cola:
            self._cola.remove(pieza)
        if pieza is not None:
            pieza.procesado = True
        self.piezas_procesadas += 1
        self.activo = bool(self._cola)
        return pieza

    def obtener_temperatura(self) -> float:
        return self.temperatura

    def __repr__(self) -> str:
        return (
            f"Horno(nombre={self.nombre!r}, piezas={self.piezas}, "
            f"temperatura={self.temperatura}, estado={self.estado})"
        )
