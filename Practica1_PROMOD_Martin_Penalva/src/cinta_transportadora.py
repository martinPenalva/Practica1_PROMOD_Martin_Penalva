"""Representa una cinta transportadora de la planta."""

from __future__ import annotations

from typing import List


class CintaTransportadora:
    """Encapsula el estado y la lógica de una cinta del circuito."""

    def __init__(self, nombre: str, velocidad: float = 0.55, direccion: tuple[float, float] = (1.0, 0.0), activa: bool = True):
        self.nombre = nombre
        self.id = nombre
        self.velocidad = float(velocidad)
        self.direccion = tuple(direccion)
        if len(self.direccion) != 2:
            raise ValueError("La dirección debe contener dx y dy.")
        self.activa = bool(activa)
        self.objetos: List[object] = []

    def iniciar(self) -> None:
        self.activa = True

    def arrancar(self) -> None:
        self.iniciar()

    def detener(self) -> None:
        self.activa = False

    def conmutar(self) -> bool:
        self.activa = not self.activa
        return self.activa

    def esta_activa(self) -> bool:
        return self.activa

    def agregar_objeto(self, objeto) -> None:
        if objeto not in self.objetos:
            self.objetos.append(objeto)

    def quitar_objeto(self, objeto) -> None:
        if objeto in self.objetos:
            self.objetos.remove(objeto)

    def actualizar(self, dt: float = 0.1) -> None:
        if dt < 0:
            raise ValueError("El intervalo de actualización no puede ser negativo.")
        if not self.activa:
            return

        dx, dy = self.direccion
        for objeto in self.objetos:
            objeto.actualizar((dx * self.velocidad * dt, dy * self.velocidad * dt, 0.0))

    def mover_objetos(self) -> None:
        self.actualizar()

    def __repr__(self) -> str:
        estado = "ACTIVA" if self.activa else "PARADA"
        return f"CintaTransportadora(nombre={self.nombre!r}, velocidad={self.velocidad}, estado={estado})"
