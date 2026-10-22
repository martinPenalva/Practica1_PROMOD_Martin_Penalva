"""Responsable de capturar imágenes y detectar piezas."""

from __future__ import annotations

import cv2
import numpy as np


class CamaraInspeccion:
    """Modela una cámara de inspección asociada a una cinta."""

    def __init__(self, nombre: str, cinta, umbral: int = 60, resolucion: tuple[int, int] = (640, 480)):
        self.nombre = nombre
        self.id = nombre
        self.cinta = cinta
        self.umbral = int(umbral)
        self.resolucion = tuple(resolucion)
        self.objeto_detectado = False
        self.forma_detectada = None
        self.ultima_imagen = None
        self.ultimo_objeto = None
        self._objetos_inspeccionados: set[str] = set()

    def capturar(self, renderer, data=None):
        if data is not None:
            renderer.update_scene(data, camera=self.nombre)
            imagen = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        else:
            imagen = renderer
        self.ultima_imagen = imagen
        return imagen

    def detectar_forma(self, imagen):
        if imagen is None:
            return None

        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, (0, 70, self.umbral), (179, 255, 255))
        alto, ancho = mascara.shape
        margen_x = max(4, ancho // 12)
        margen_y = max(4, alto // 12)
        mascara[:margen_y, :] = 0
        mascara[-margen_y:, :] = 0
        mascara[:, :margen_x] = 0
        mascara[:, -margen_x:] = 0
        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidatos = []
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area < 90 or area > 26000:
                continue
            x, y, ancho_objeto, alto_objeto = cv2.boundingRect(contorno)
            if ancho_objeto < 8 or alto_objeto < 8:
                continue
            candidatos.append((area, contorno, ancho_objeto, alto_objeto))
        if not candidatos:
            return None

        area, contorno, ancho_objeto, alto_objeto = max(candidatos, key=lambda item: item[0])
        perimetro = cv2.arcLength(contorno, True)
        if perimetro <= 0:
            return None
        circularidad = 4 * np.pi * area / (perimetro * perimetro)
        if circularidad > 0.82:
            return "circulo"

        aproximacion = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)
        vertices = len(aproximacion)
        if vertices == 3:
            return "triangulo"
        if vertices == 4:
            proporcion = max(ancho_objeto, alto_objeto) / min(ancho_objeto, alto_objeto)
            return "rectangulo" if proporcion > 1.2 else "cuadrado"
        if vertices > 4:
            return "circulo"
        return "desconocido"

    def inspeccionar(self, objeto):
        """Inspecciona una pieza del modelo lógico y detiene su cinta."""
        if objeto is None:
            self.objeto_detectado = False
            self.forma_detectada = None
            self.ultimo_objeto = None
            return None

        self.objeto_detectado = True
        self.forma_detectada = objeto.forma
        self.ultimo_objeto = objeto
        self.cinta.detener()
        self._objetos_inspeccionados.add(objeto.id)
        return self.forma_detectada

    def procesar(self, imagen=None, detener_cinta: bool = True):
        if imagen is None:
            imagen = self.ultima_imagen
        self.forma_detectada = self.detectar_forma(imagen)
        self.objeto_detectado = self.forma_detectada is not None
        if self.objeto_detectado and detener_cinta:
            self.cinta.detener()
        return self.forma_detectada

    def actualizar(self, renderer, data=None, detener_cinta: bool = True):
        if renderer is None:
            return None
        imagen_capturada = self.capturar(renderer, data)
        self.procesar(imagen_capturada, detener_cinta=detener_cinta)
        return imagen_capturada

    def __repr__(self) -> str:
        return f"CamaraInspeccion(nombre={self.nombre!r}, cinta={self.cinta.nombre!r}, detectado={self.objeto_detectado})"
