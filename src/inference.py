
"""
GeoAI Land Cover Classifier
Módulo de inferencia.

Este módulo contiene la lógica necesaria para convertir
una imagen RGB de EuroSAT en características HOG y
utilizar un modelo entrenado para clasificarla.

El entrenamiento se realiza en los notebooks.
Aquí únicamente realizamos predicciones.
"""

import numpy as np

from skimage.color import rgb2gray
from skimage.feature import hog


# ------------------------------------------------------------
# CONFIGURACIÓN DE LAS IMÁGENES
# ------------------------------------------------------------

# EuroSAT RGB contiene imágenes de 64 x 64 píxeles.
# Nuestro modelo se entrenó utilizando ese tamaño.

TAMANO_IMAGEN = (64, 64, 3)


# ------------------------------------------------------------
# EXTRACCIÓN DE CARACTERÍSTICAS HOG
# ------------------------------------------------------------

def extraer_hog(imagen_rgb):
    """
    Convierte una imagen RGB en características HOG.

    Parámetros
    ----------
    imagen_rgb:
        Imagen RGB representada como array NumPy
        con dimensiones (64, 64, 3).

    Devuelve
    --------
    np.ndarray:
        Vector de 1764 características HOG.
    """

    # Convertimos la imagen a un array NumPy.
    imagen = np.asarray(imagen_rgb)

    # Verificamos que el tamaño sea compatible
    # con el utilizado durante el entrenamiento.
    if imagen.shape != TAMANO_IMAGEN:
        raise ValueError(
            f"Dimensiones incorrectas: {imagen.shape}. "
            f"Se esperaba {TAMANO_IMAGEN}."
        )

    # Convertimos la imagen RGB a escala de grises.
    imagen_gris = rgb2gray(imagen)

    # Extraemos los descriptores HOG utilizando
    # exactamente la configuración del entrenamiento.
    caracteristicas = hog(
        imagen_gris,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=False,
        visualize=False,
        feature_vector=True
    )

    return caracteristicas


# ------------------------------------------------------------
# PREDICCIÓN DE COBERTURA DEL SUELO
# ------------------------------------------------------------

def predecir_cobertura(imagen_rgb, modelo):
    """
    Predice la categoría de cobertura del suelo.

    Parámetros
    ----------
    imagen_rgb:
        Imagen RGB de dimensiones (64, 64, 3).

    modelo:
        Pipeline entrenado con scikit-learn.

    Devuelve
    --------
    int:
        Identificador numérico de la clase predicha.
    """

    # Obtenemos las características HOG de la imagen.
    caracteristicas = extraer_hog(imagen_rgb)

    # Transformamos el vector de 1 dimensión
    # en una matriz con una fila.
    #
    # Antes: (1764,)
    # Después: (1, 1764)
    caracteristicas = caracteristicas.reshape(1, -1)

    # El pipeline aplica StandardScaler y después
    # utiliza SGDClassifier para predecir.
    prediccion = modelo.predict(caracteristicas)

    # Devolvemos el identificador de la clase.
    return int(prediccion[0])
