
"""
GeoAI Land Cover Classifier
---------------------------
Inferencia por lotes y evaluación de muestras.

Este programa:
1. Recupera un modelo de Machine Learning entrenado.
2. Lee imágenes RGB de la carpeta samples/.
3. Obtiene una predicción para cada imagen compatible.
4. Compara la predicción con la etiqueta real, si existe.
5. Guarda los resultados en un archivo CSV.

IMPORTANTE:
Los resultados de estas muestras son exploratorios.
No representan la evaluación final del modelo.
"""

# ============================================================
# 1. IMPORTAR LIBRERÍAS Y FUNCIONES
# ============================================================

import csv
from pathlib import Path

from PIL import Image

# Reutilizamos nuestro módulo de inferencia.
# Aquí no entrenamos el modelo.
from src.inference import cargar_modelo, predecir_cobertura


# ============================================================
# 2. CONFIGURAR RUTAS
# ============================================================

# Localizamos la carpeta principal del proyecto.
CARPETA_PROYECTO = Path(__file__).resolve().parent

# Modelo previamente entrenado.
RUTA_MODELO = (
    CARPETA_PROYECTO
    / "models"
    / "land_cover_hog_sgd.joblib"
)

# Carpeta de fotografías.
CARPETA_IMAGENES = CARPETA_PROYECTO / "samples"

# Archivo con las categorías reales.
RUTA_ETIQUETAS = CARPETA_IMAGENES / "labels.csv"

# Archivo en el que guardaremos las predicciones.
RUTA_RESULTADOS = (
    CARPETA_PROYECTO
    / "results"
    / "sample_predictions.csv"
)


# ============================================================
# 3. DEFINIR LAS CLASES DE EUROSAT
# ============================================================

# El orden debe coincidir con el utilizado al entrenar.

CLASES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


# ============================================================
# 4. CARGAR LAS ETIQUETAS REALES
# ============================================================

# Creamos un diccionario:
#
# {
#     "test_image.png": "AnnualCrop",
#     "test_image_1.png": "Residential",
#     ...
# }
#
# Esto nos permite consultar rápidamente la
# categoría real de cada fotografía.

with open(
    RUTA_ETIQUETAS,
    "r",
    encoding="utf-8-sig",
    newline=""
) as archivo:

    lector = csv.DictReader(archivo)

    etiquetas_reales = {
        fila["filename"]: fila["true_label"]
        for fila in lector
    }


# ============================================================
# 5. CARGAR EL MODELO
# ============================================================

# Recuperamos el pipeline guardado con joblib.
# No volvemos a entrenarlo.

modelo = cargar_modelo(RUTA_MODELO)

print("Modelo cargado correctamente.")


# ============================================================
# 6. LOCALIZAR LAS IMÁGENES
# ============================================================

# Seleccionamos únicamente archivos de imagen.
# sorted() garantiza un orden consistente.

imagenes = sorted(
    ruta
    for ruta in CARPETA_IMAGENES.iterdir()
    if ruta.is_file()
    and ruta.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

print(f"Imágenes encontradas: {len(imagenes)}")


# ============================================================
# 7. CLASIFICAR Y EVALUAR EN UN SOLO RECORRIDO
# ============================================================

# Esta lista almacenará los resultados.
# Cada elemento será un diccionario con los datos
# correspondientes a una imagen.

resultados = []

for ruta_imagen in imagenes:

    # Abrimos la fotografía y garantizamos formato RGB.
    with Image.open(ruta_imagen) as archivo:
        imagen = archivo.convert("RGB")

    # El modelo solo admite imágenes de 64 x 64.
    if imagen.size != (64, 64):
        print(f"Omitida por tamaño: {ruta_imagen.name}")
        continue

    # Obtenemos la predicción numérica.
    indice_predicho = predecir_cobertura(
        imagen,
        modelo
    )

    # Convertimos el número en una categoría.
    prediccion = CLASES[indice_predicho]

    # Buscamos la categoría real.
    # Si no existe, devolvemos None.
    realidad = etiquetas_reales.get(ruta_imagen.name)

    # Solo podemos evaluar cuando conocemos
    # la categoría correcta.
    if realidad is None:
        correcto = None
        estado = "SIN ETIQUETA"
    else:
        correcto = prediccion == realidad
        estado = "ACIERTO" if correcto else "ERROR"

    # Guardamos la información de esta imagen.
    resultados.append({
        "filename": ruta_imagen.name,
        "true_label": realidad or "",
        "predicted_label": prediccion,
        "correct": correcto if correcto is not None else "",
    })

    # Mostramos el resultado en la terminal.
    print()
    print("Imagen:", ruta_imagen.name)
    print("Categoría real:", realidad or "Desconocida")
    print("Predicción:", prediccion)
    print("Resultado:", estado)


# ============================================================
# 8. CALCULAR EL RESUMEN
# ============================================================

# Seleccionamos únicamente las imágenes
# que tienen una etiqueta real conocida.

evaluados = [
    resultado
    for resultado in resultados
    if resultado["correct"] != ""
]

# Contamos los aciertos.
aciertos = sum(
    resultado["correct"] is True
    for resultado in evaluados
)

print()
print("=" * 45)
print("RESUMEN EXPLORATORIO")
print("=" * 45)

if evaluados:

    accuracy = aciertos / len(evaluados)

    print("Imágenes evaluadas:", len(evaluados))
    print("Aciertos:", aciertos)
    print("Errores:", len(evaluados) - aciertos)
    print(f"Accuracy: {accuracy:.2%}")

else:
    print("No hay imágenes etiquetadas para evaluar.")


# ============================================================
# 9. GUARDAR LOS RESULTADOS EN CSV
# ============================================================

# Creamos la carpeta results/ si no existe.
RUTA_RESULTADOS.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Definimos las columnas de nuestro CSV.
columnas = [
    "filename",
    "true_label",
    "predicted_label",
    "correct",
]

# Escribimos la tabla de resultados.
# Si el archivo ya existe, se actualiza con
# los resultados de la ejecución actual.

with open(
    RUTA_RESULTADOS,
    "w",
    encoding="utf-8",
    newline=""
) as archivo:

    escritor = csv.DictWriter(
        archivo,
        fieldnames=columnas
    )

    # Escribimos los encabezados.
    escritor.writeheader()

    # Escribimos una fila por fotografía.
    escritor.writerows(resultados)

print()
print("Resultados guardados en:")
print(RUTA_RESULTADOS)
