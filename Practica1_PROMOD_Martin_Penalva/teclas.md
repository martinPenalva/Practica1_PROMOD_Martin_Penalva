# Guía visual de teclas

Esta guía explica cómo probar la planta desde sus ventanas de MuJoCo y OpenCV.

## Iniciar la simulación

Desde la carpeta raíz del proyecto ejecuta:

```bash
python main.py
```

Se abrirá el visor 3D de MuJoCo y la ventana OpenCV con las cámaras de la
planta. Para que se reciban las teclas, haz clic primero en la ventana de
cámaras. Las teclas se escriben sin mantener `Shift`.

## Controles

| Tecla | Acción | Qué comprobar visualmente |
| --- | --- | --- |
| `1` | Arranca o detiene la cinta 1. | El indicador `C1` del mosaico cambia entre `ON` y `OFF`; las demás cintas no deberían cambiar de estado. |
| `2` | Arranca o detiene la cinta 2. | Comprueba el indicador `C2` y que las otras cintas mantienen su estado. |
| `3` | Arranca o detiene la cinta 3. | Comprueba el indicador `C3` y que las otras cintas mantienen su estado. |
| `4` | Arranca o detiene la cinta 4. | Comprueba el indicador `C4` y que las otras cintas mantienen su estado. |
| `5` | Arranca o detiene la cinta 5, la que lleva al horno. | Comprueba el indicador `C5` y el movimiento en la cámara del horno. |
| `P` | Pausa o reanuda la simulación. | El movimiento del modelo se detiene y continúa al pulsar de nuevo. |
| `U` | Aumenta la temperatura del horno en 40 °C. | La consola muestra la temperatura y el color; el horno cambia de azul a naranja y luego a rojo. |
| `D` | Disminuye la temperatura del horno en 40 °C. | La temperatura y el color bajan; el mínimo es 15 °C. |
| `T` | Vuelve a enviar la pose objetivo configurada para la prueba de IK. | Observa el movimiento del brazo Panda en el visor 3D. Solo funciona con el modo de prueba IK activo. |
| `H` | Envía el brazo Panda a la pose cartesiana HOME. | Observa el movimiento del brazo en el visor 3D. Solo funciona con el modo de prueba IK activo. |
| `R` | Reinicia las piezas, el robot, el horno y las cintas. | Las piezas vuelven a su disposición inicial y la temperatura del horno vuelve a 20 °C. |
| `S` | Guarda una captura del mosaico actual. | La consola indica dónde se guardó el PNG, dentro de `capturas/`. |
| `Q` o `Esc` | Cierra la simulación. | Se cierran las ventanas de cámaras y de MuJoCo. |

## Prueba visual del color del horno

La temperatura inicial es 20 °C. Pulsa `U` y observa el horno en el visor 3D:

1. Con 20 °C, el color es azul.
2. Tras dos pulsaciones de `U` (100 °C), debe ser naranja.
3. Tras dos pulsaciones más (180 °C), debe ser rojo.
4. Pulsa `D` para reducir la temperatura y comprobar el cambio inverso.

Cada pulsación aumenta o disminuye 40 °C. El valor está limitado al intervalo
de 15 °C a 250 °C, y la consola muestra la temperatura y el color actuales.

## Prueba de cintas y cámaras

1. Con la simulación en marcha, pulsa `1` y verifica que solo cambia el estado
   de la cinta 1; pulsa `1` otra vez para reanudarla.
2. Repite con las teclas `2` a `5`.
3. Observa los paneles `CINTA 1` a `CINTA 5`: muestran la imagen de cada cámara
   y el resultado de la detección cuando hay una forma reconocida.
4. Si una cámara detecta una pieza, la cinta asociada se detiene brevemente y
   puede reanudarse automáticamente. Una parada manual con su tecla numérica
   se mantiene hasta volver a pulsar esa tecla.
