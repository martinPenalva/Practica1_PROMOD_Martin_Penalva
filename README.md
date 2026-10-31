# Planta industrial orientada a objetos

La aplicación principal ejecuta la planta física en MuJoCo, muestra las cinco
cámaras de inspección mediante OpenCV y utiliza el controlador de cinemática
inversa del Franka Panda que ya proporciona el proyecto.

El proyecto y todos sus recursos están dentro de
`Practica1_PROMOD_Martin_Penalva/`. Ejecuta los siguientes comandos desde la
raíz del repositorio.

## Instalación

```bash
git clone --recurse-submodules https://github.com/martinPenalva/Practica1_PROMOD_Martin_Penalva.git
cd Practica1_PROMOD_Martin_Penalva
python -m pip install -r requirements.txt
```

El modelo de Franka se encuentra en `vendor/mujoco_menagerie`. También se puede
indicar otra instalación mediante `MUJOCO_MENAGERIE_PATH`.

## Ejecución

```bash
cd Practica1_PROMOD_Martin_Penalva
python main.py
```

Controles de la ventana de cámaras:

- `Q` o `Esc`: salir.
- `P`: pausar o reanudar.
- `1`–`5`: detener o reanudar individualmente una cinta.
- `R`: reiniciar piezas, robot, horno y cintas.
- `S`: guardar una captura del mosaico en `capturas/`.
- `T` y `H`: reenviar la pose IK de prueba o volver a la pose HOME.
- `U`: subir la temperatura del horno 40 °C.
- `D`: bajar la temperatura del horno 40 °C.

La cámara toma imágenes del modelo MuJoCo y detecta formas con OpenCV. Al
detectar una pieza, activa `objeto_detectado` y la planta detiene brevemente la
cinta asociada antes de reanudarla, salvo que se haya detenido manualmente. El
horno acumula temperatura, procesa las piezas que llegan a la salida y cambia
el color de su geometría en el modelo según la temperatura.

## Organización

### Código Python de la aplicación

- `main.py`: punto de entrada; crea la planta y arranca su ciclo de ejecución.
- `src/__init__.py`: declara el paquete y exporta las clases principales.
- `src/planta_industrial.py`: configura y coordina MuJoCo, cintas, cámaras,
  piezas, horno y robot durante la simulación.
- `src/cinta_transportadora.py`: encapsula el estado, la velocidad y el
  movimiento de los objetos de cada cinta.
- `src/camara_inspeccion.py`: captura imágenes, detecta y clasifica formas, y
  comunica el estado de inspección a la cinta asociada.
- `src/objeto_industrial.py`: define las piezas, su identidad, forma, posición y
  segmento; incluye las variantes `Cilindro` y `Caja`.
- `src/horno.py`: gestiona la temperatura, la cola de piezas, su procesamiento
  y el color de la geometría del horno en MuJoCo.
- `src/RobotPanda.py`: adapta el controlador IK existente para que la planta
  pueda inicializar y controlar el Panda como objeto.
- `planta_4_cintas.py`: conserva la configuración y las funciones procedurales
  de la simulación original, reutilizadas para la física, el transporte y la
  representación de la planta.
- `robot.py`: implementa el controlador procedural del Panda y la cinemática
  inversa que reutiliza `src/RobotPanda.py`.

### Scripts Python incluidos con MuJoCo Menagerie

Estos scripts son herramientas del repositorio incluido en `vendor/`; no forman
parte del arranque normal de la aplicación:

- `vendor/mujoco_menagerie/build_registry.py`: comprueba los modelos MJCF y
  genera el registro de modelos de Menagerie.
- `vendor/mujoco_menagerie/catalog.py`: mantiene metadatos y utilidades del
  catálogo de robots, como nombres, vistas previas y licencias.
- `vendor/mujoco_menagerie/format_xml.py`: formatea archivos XML/MJCF.
- `vendor/mujoco_menagerie/generate_gallery.py`: genera imágenes de vista previa
  y actualiza la galería del README de Menagerie.
- `vendor/mujoco_menagerie/regenerate_license.py`: regenera el archivo de
  licencias de Menagerie a partir de las licencias de sus modelos.

Otros recursos principales: `src/planta.xml` define la geometría y las cámaras
MuJoCo; `requirements.txt` recoge las dependencias; y `vendor/mujoco_menagerie/`
contiene el modelo Panda y sus recursos.

##  Pruebas mínimas

No hay una suite automatizada de pruebas en el proyecto. Para la comprobación
visual, desde la carpeta raíz instala los requisitos y ejecuta:

```bash
cd Practica1_PROMOD_Martin_Penalva
python -m pip install -r requirements.txt
python main.py
```

Comprueba lo siguiente:

1. **Arranque de MuJoCo.** Deben abrirse el visor 3D y la ventana de cámaras; la
   consola muestra que se ha cargado el modelo del Panda.
2. **Arranque y parada de cintas.** En la ventana de cámaras, pulsa `1` y
   comprueba que la Cinta 1 se detiene; púlsalo de nuevo para arrancarla.
   Repite con `2`–`5`.
3. **Estado independiente.** Con varias cintas en marcha, detén una y confirma
   en el mosaico que las demás siguen funcionando. La línea de estado muestra
   `C1:ON/OFF` hasta `C5:ON/OFF`.
4. **Captura de las cámaras.** Comprueba que los paneles `CINTA 1`–`CINTA 5`
   muestran la vista correspondiente y que no aparecen errores del renderer.
5. **Visualización OpenCV.** Comprueba que la ventana muestra el mosaico de las
   cinco cámaras y la vista general. Pulsa `S` para guardar una captura en
   `capturas/`.
6. **Detección de formas.** Observa el texto de cada panel al pasar una pieza:
   OpenCV clasifica contornos como `circulo`, `triangulo`, `cuadrado` o
   `rectangulo`. En la comprobación realizada sobre la escena inicial, la
   cámara 4 detectó un círculo; las otras formas deben comprobarse cuando una
   pieza adecuada quede dentro del campo de visión.
7. **Variable booleana.** Al aparecer `PIEZA DETECTADA` (o el nombre de la
   forma) en un panel, la cámara correspondiente pone
   `objeto_detectado = True`; cuando deja de encontrar un contorno válido,
   vuelve a `False`.
8. **Parada por detección.** Al detectar una pieza, la cinta asociada se
   detiene para inspeccionarla. Actualmente la parada es temporal: tras
   `0,35 s` de simulación la cinta puede arrancar aunque
   `objeto_detectado` continúe en `True`. Por tanto, la parada inicial funciona,
   pero no se mantiene durante todo el tiempo que la variable siga siendo
   verdadera.
9. **Reanudación controlada.** La lógica establece una espera de `0,35 s` y
   respeta las paradas manuales antes de reanudar automáticamente. La
   reanudación no espera a que desaparezca la detección; si el requisito exige
   reanudar únicamente después de que `objeto_detectado` vuelva a `False`, esta
   prueba no queda satisfecha con el comportamiento actual. Para terminar,
   pulsa `Q` o `Esc`.
10. **Cambio de temperatura.** Durante la simulación, pulsa `U` para subir
    40 °C o `D` para bajar 40 °C. La temperatura y el color actual también se
    imprimen en la consola.
11. **Color del horno.** El color pasa de azul a naranja a partir de 80 °C y a
    rojo a partir de 180 °C. Desde los 20 °C iniciales, pulsa `U` dos veces
    para llegar a 100 °C (naranja) y otras dos veces para llegar a 180 °C
    (rojo). Pulsa `D` para observar el cambio en sentido inverso. Los límites
    del horno son 15 °C y 250 °C.
12. **IK del robot.** El modo de prueba IK está activo por defecto. En la ventana
    de cámaras, pulsa `T` para enviar la pose configurada y `H` para volver a
    la pose HOME; confirma en el visor 3D que el Panda se mueve.

Se han comprobado también mediante una ejecución sin ventanas la inicialización
y un paso de MuJoCo, el estado independiente de las cintas, la captura de las
cinco cámaras, el mosaico OpenCV, detecciones de círculo/cuadrado/rectángulo en
imágenes de prueba, el cambio de color del horno y la llamada al controlador IK
existente. Esto no sustituye las comprobaciones visuales de la ventana ni
confirma la detección de todas las formas en movimiento.
