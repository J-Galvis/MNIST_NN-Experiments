import numpy as np
import os
import json
import pickle
from datetime import datetime


# ── Carpeta donde se guardan los modelos ─────────────────────────────────────
CARPETA_MODELOS = os.path.join(os.path.dirname(__file__), '../modelos_guardados')


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(W1, b1, W2, b2, nombre_modelo, precision_test=None,
                   epocas=None, learning_rate=None, training_time=None, info_extra=None):
    """
    Guarda los pesos de una red neuronal entrenada en un archivo .pkl.

    """
    # Crear la carpeta si no existe
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    # Construir metadatos
    metadatos = {
        'nombre_modelo': nombre_modelo,
        'fecha_guardado': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'precision_test': precision_test,
        'epocas': epocas,
        'learning_rate': learning_rate,
        'training_time_seconds': training_time,
        'arquitectura': {
            'entrada': int(W1.shape[0]),
            'oculta': int(W1.shape[1]),
            'salida': int(W2.shape[1])
        }
    }
    if info_extra:
        metadatos['info_extra'] = info_extra

    # Nombre del archivo: nombre_modelo + timestamp para evitar sobreescrituras
    nombre_archivo = f"{nombre_modelo}.pkl"
    ruta_archivo = os.path.join(CARPETA_MODELOS, nombre_archivo)

    # Empaquetar los datos a guardar en un diccionario
    datos_modelo = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'metadatos': metadatos
    }

    # Guardar con pickle
    with open(ruta_archivo, 'wb') as archivo:
        pickle.dump(datos_modelo, archivo)

    return ruta_archivo


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo(nombre_modelo=None, ruta_archivo=None):
    """
    Carga un modelo previamente guardado desde un archivo .pkl.
    """
    if ruta_archivo is None:
        # Buscar en la carpeta de modelos
        if not os.path.exists(CARPETA_MODELOS):
            raise FileNotFoundError(
                f"No se encontró la carpeta de modelos: {CARPETA_MODELOS}\n"
                f"Primero entrena y guarda un modelo."
            )

        # Listar todos los archivos .pkl
        archivos = [f for f in os.listdir(CARPETA_MODELOS) if f.endswith('.pkl')]

        if not archivos:
            raise FileNotFoundError(
                f"No hay modelos guardados en: {CARPETA_MODELOS}"
            )

        # Filtrar por nombre si se proporcionó
        if nombre_modelo:
            archivos = [f for f in archivos if nombre_modelo in f]
            if not archivos:
                raise FileNotFoundError(
                    f"No se encontró ningún modelo con nombre '{nombre_modelo}' "
                    f"en {CARPETA_MODELOS}"
                )

        # Tomar el más reciente (ordenar por nombre que incluye timestamp)
        archivos.sort()
        archivo_seleccionado = archivos[-1]
        ruta_archivo = os.path.join(CARPETA_MODELOS, archivo_seleccionado)

    # Cargar el archivo .pkl
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    with open(ruta_archivo, 'rb') as archivo:
        datos = pickle.load(archivo)

    W1 = datos['W1']
    b1 = datos['b1']
    W2 = datos['W2']
    b2 = datos['b2']
    metadatos = datos['metadatos']

    return W1, b1, W2, b2, metadatos
