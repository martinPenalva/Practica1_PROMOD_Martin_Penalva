"""Representa las piezas que circulan por la planta."""

from __future__ import annotations


class ObjetoIndustrial:
    """Pieza con identidad, forma, posición y segmento actual."""

    def __init__(
        self,
        identificador: str,
        forma: str,
        segmento: int = 0,
        posicion: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        self.id = identificador
        self.identificador = identificador
        self.tipo = forma
        self.forma = forma
        self.segmento = int(segmento)
        self.posicion = [0.0, 0.0, 0.0] if posicion is None else list(posicion)
        if len(self.posicion) != 3:
            raise ValueError("La posición debe contener x, y y z.")
        self.orientacion = [1.0, 0.0, 0.0, 0.0]
        self.procesado = False

    def actualizar(self, desplazamiento: tuple[float, float, float] | list[float]) -> None:
        if len(desplazamiento) != 3:
            raise ValueError("El desplazamiento debe contener dx, dy y dz.")
        for eje, delta in enumerate(desplazamiento):
            self.posicion[eje] += float(delta)

    def mover(self, desplazamiento: tuple[float, float, float] | list[float]) -> None:
        self.actualizar(desplazamiento)

    def cambiar_segmento(self, segmento: int) -> None:
        self.segmento = int(segmento)

    def obtener_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return tuple(self.posicion), tuple(self.orientacion)

    def __repr__(self) -> str:
        return (
            f"ObjetoIndustrial(id={self.id!r}, forma={self.forma!r}, "
            f"segmento={self.segmento}, posicion={self.posicion})"
        )


class Cilindro(ObjetoIndustrial):
    def __init__(
        self,
        identificador: str,
        segmento: int = 0,
        posicion: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        super().__init__(identificador, "cilindro", segmento, posicion)


class Caja(ObjetoIndustrial):
    def __init__(
        self,
        identificador: str,
        segmento: int = 0,
        posicion: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        super().__init__(identificador, "cubo", segmento, posicion)
