import cv2


class FiguraDetectada:
    """
    Representa una figura encontrada en la imagen.

    Guarda la información necesaria para que los módulos
    posteriores puedan analizar y clasificar la figura.
    """

    def __init__(
        self,
        mascara,
        x,
        y,
        ancho,
        alto,
        area,
        centroide
    ):
        # Máscara binaria correspondiente únicamente a esta figura.
        self.mascara = mascara

        # Coordenadas de la esquina superior izquierda
        # del rectángulo que contiene la figura.
        self.x = x
        self.y = y

        # Dimensiones del rectángulo delimitador.
        self.ancho = ancho
        self.alto = alto

        # Cantidad de píxeles que pertenecen a la figura.
        self.area = area

        # Coordenadas (x, y) del centro de la figura.
        self.centroide = centroide


def detectar_figuras(mascara):
    """
    Detecta las figuras presentes en una máscara binaria.

    Cada región conectada de la máscara se considera una
    figura independiente.

    Args:
        mascara: Imagen binaria donde las figuras están
                 representadas por píxeles blancos y el
                 fondo por píxeles negros.

    Returns:
        Lista de objetos FiguraDetectada, uno por cada
        figura encontrada.
    """

    # Detectar componentes conectados en la máscara.
    #
    # num_labels: cantidad total de regiones encontradas.
    # labels: matriz que indica a qué región pertenece
    #         cada píxel.
    # stats: información geométrica de cada región.
    # centroids: centroide de cada región.
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8
        )
    )

    figuras = []

    # La etiqueta 0 corresponde al fondo, por eso
    # comenzamos desde 1.
    for i in range(1, num_labels):

        # Obtener la posición de la figura.
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        # Obtener las dimensiones de su rectángulo delimitador.
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]

        # Obtener el área de la figura en píxeles.
        area = stats[i, cv2.CC_STAT_AREA]

        # Convertir el centroide a una tupla (x, y).
        centroide = tuple(centroids[i])

        # Crear una máscara que contenga únicamente
        # la figura actual.
        mascara_figura = (
            labels == i
        ).astype("uint8")

        # Crear el objeto que representa la figura.
        figura = FiguraDetectada(
            mascara_figura,
            x,
            y,
            ancho,
            alto,
            area,
            centroide
        )

        figuras.append(figura)

    return figuras