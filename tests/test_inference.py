
"""
Pruebas automáticas del módulo de inferencia.

Objetivo:
Comprobar que la extracción de características HOG
produce los resultados estructurales esperados.

Estas pruebas NO miden la precisión del modelo.
"""

import unittest
import numpy as np

from src.inference import extraer_hog


class TestExtraccionHOG(unittest.TestCase):
    """
    Agrupa las pruebas de extracción de características.
    """

    def test_imagen_valida(self):
        """
        Una imagen RGB de 64x64 debe producir
        un vector de 1764 características HOG.
        """

        # Creamos una imagen artificial totalmente negra.
        #
        # No necesitamos descargar EuroSAT porque
        # solo comprobamos el funcionamiento del código.

        imagen = np.zeros(
            (64, 64, 3),
            dtype=np.uint8
        )

        # Ejecutamos nuestra función real.
        caracteristicas = extraer_hog(imagen)

        # Comprobamos las dimensiones del resultado.
        self.assertEqual(
            caracteristicas.shape,
            (1764,)
        )

        # Comprobamos que los valores son finitos:
        # no deben aparecer NaN ni infinito.
        self.assertTrue(
            np.isfinite(caracteristicas).all()
        )

    def test_imagen_incorrecta(self):
        """
        Una imagen con dimensiones incompatibles
        debe provocar un ValueError.
        """

        # Creamos una imagen de tamaño incorrecto.
        imagen = np.zeros(
            (128, 128, 3),
            dtype=np.uint8
        )

        # Verificamos que nuestra función
        # rechaza la imagen correctamente.
        with self.assertRaises(ValueError):
            extraer_hog(imagen)


# Permite ejecutar las pruebas directamente
# mediante: python -m unittest
if __name__ == "__main__":
    unittest.main()
