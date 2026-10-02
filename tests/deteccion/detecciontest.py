from pathlib import Path

import cv2
import numpy as np

from preprocesamiento.preprocess import preprocesar
from deteccion.detector import detectar_figuras


BASE_DIR = Path(__file__).resolve().parents[2]

ruta = BASE_DIR / "dataset_figuras" / "02_cuadrado.bmp"


# Preprocesamiento

datos = preprocesar(ruta)


# Deteccion

figuras = detectar_figuras(datos.mascara)


print("Número de figuras:", len(figuras))


for i, figura in enumerate(figuras, start=1):

    print()
    print(f"Figura {i}")
    print("Posición:", figura.x, figura.y)
    print(
        "Tamaño:",
        figura.ancho,
        "x",
        figura.alto
    )
    print("Área:", figura.area)
    print("Centroide:", figura.centroide)


# Vizualización 

imagen_deteccion = np.zeros_like(datos.mascara)

for i, figura in enumerate(figuras, start=1):

    imagen_deteccion[
        figura.mascara == 1
    ] = i


cv2.imshow(
    "Figuras detectadas",
    imagen_deteccion.astype("uint8") * 50
)

cv2.waitKey(0)
cv2.destroyAllWindows()