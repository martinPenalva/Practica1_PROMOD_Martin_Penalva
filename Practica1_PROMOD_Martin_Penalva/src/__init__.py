"""Paquete con la refactorización orientada a objetos de la planta industrial."""

from .cinta_transportadora import CintaTransportadora
from .camara_inspeccion import CamaraInspeccion
from .objeto_industrial import ObjetoIndustrial, Cilindro, Caja
from .horno import Horno
from .RobotPanda import RobotPanda
from .planta_industrial import PlantaIndustrial

__all__ = [
    "CintaTransportadora",
    "CamaraInspeccion",
    "ObjetoIndustrial",
    "Cilindro",
    "Caja",
    "Horno",
    "RobotPanda",
    "PlantaIndustrial",
]
