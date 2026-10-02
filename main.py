from pathlib import Path

import cv2

from src.preprocesamiento.preprocess import preprocesar
from src.deteccion.detector import detectar_figuras
from src.clasificacion.clasifier import (
    clasificar_figura,
    obtener_color,
    rgb_a_hex
)


def cargar_imagen(ruta):
    """
    Valida la ruta y carga una imagen BMP.

    Args:
        ruta (Path): Ruta del archivo de imagen.

    Returns:
        Imagen cargada mediante OpenCV.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no es BMP o no puede leerse.
    """

    # Verificar que el archivo exista.
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe la imagen: {ruta} o el directorio es incorrecto"
        )

    # Verificar que sea un archivo BMP.
    if ruta.suffix.lower() != ".bmp":
        raise ValueError(
            "El archivo debe tener extensión .bmp"
        )

    # Intentar cargar la imagen.
    imagen = cv2.imread(str(ruta))

    # OpenCV devuelve None cuando no pudo leerla.
    if imagen is None:
        raise ValueError(
            f"No se pudo leer la imagen: {ruta}"
        )

    return imagen


def main():
    """
    Ejecuta el proceso completo de reconocimiento de figuras:

    1. Localiza la imagen.
    2. Carga la imagen original.
    3. Realiza el preprocesamiento.
    4. Detecta las figuras.
    5. Clasifica cada figura.
    6. Obtiene su color hexadecimal.
    7. Muestra los resultados.
    """

    
    # Ruta de la Imagen
    

    # La ruta se construye a partir de la ubicación
    # de este archivo, evitando depender del directorio
    # desde el que se ejecute el programa.
    ruta = (
        Path(__file__).resolve().parent
        / "dataset_figuras"
        / "01_circulo.bmp"
    )

    # Cargar Imagen


    imagen = cargar_imagen(ruta)


    # Preprocesamiento


    # Se genera la máscara que posteriormente
    # utilizará el detector.
    datos = preprocesar(ruta)


    # Deteccion


    # Se obtienen las figuras individuales
    # encontradas en la máscara.
    figuras = detectar_figuras(
        datos.mascara
    )


    # Informacion General
    

    print()
    print("                                        ")
    print("       RECONOCIMIENTO DE FIGURAS        ")
    print("                                        ")
    print()

    print(f"Imagen: {ruta.name}")
    print(f"Figuras detectadas: {len(figuras)}")
    print()

    
    # Clasificar Figuras


    for i, figura in enumerate(figuras, start=1):

        # Determinar el tipo de figura:
        # T, C, O o X.
        tipo = clasificar_figura(figura)

        # Obtener el color de los píxeles
        # pertenecientes a la figura.
        color_rgb = obtener_color(
            imagen,
            figura.mascara
        )

        # Convertir el color RGB a hexadecimal.
        color_hex = rgb_a_hex(
            color_rgb
        )

        # Mostrar el resultado final.
        print(
            f"Figura {i} → "
            f"{color_hex} → "
            f"{tipo}"
        )

    print()



# Punto de Entrada


if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print(f"\nError: {error}")

    except ValueError as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(f"\nError inesperado: {error}")