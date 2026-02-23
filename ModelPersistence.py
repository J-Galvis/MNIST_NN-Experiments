import numpy as np
import os
import json
from datetime import datetime


# ── Carpeta donde se guardan los modelos ─────────────────────────────────────
CARPETA_MODELOS = os.path.join(os.path.dirname(__file__), 'modelos_guardados')


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(W1, b1, W2, b2, nombre_modelo, precision_test=None,
                   epocas=None, learning_rate=None, info_extra=None):
    """
    Guarda los pesos de una red neuronal entrenada en un archivo .npz.

    Parámetros
    ──────────
    W1, b1, W2, b2 : np.array
        Pesos y sesgos entrenados de la red neuronal.
    nombre_modelo : str
        Nombre descriptivo del modelo (ej: 'BasicNN', 'Arnovi_5p', 'Diego_5p').
        Se usa como nombre del archivo.
    precision_test : float, opcional
        Precisión final en el dataset de test (ej: 92.5).
    epocas : int, opcional
        Número de épocas de entrenamiento.
    learning_rate : float, opcional
        Tasa de aprendizaje usada.
    info_extra : dict, opcional
        Cualquier información adicional que se quiera guardar.

    Retorna
    ───────
    ruta_archivo : str
        Ruta completa del archivo guardado.
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
        'arquitectura': {
            'entrada': int(W1.shape[0]),
            'oculta': int(W1.shape[1]),
            'salida': int(W2.shape[1])
        }
    }
    if info_extra:
        metadatos['info_extra'] = info_extra

    # Convertir metadatos a string JSON para guardarlo en el .npz
    metadatos_json = json.dumps(metadatos, ensure_ascii=False)

    # Nombre del archivo: nombre_modelo + timestamp para evitar sobreescrituras
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    nombre_archivo = f"{nombre_modelo}_{timestamp}.npz"
    ruta_archivo = os.path.join(CARPETA_MODELOS, nombre_archivo)

    # Guardar con np.savez_compressed (más eficiente en espacio)
    np.savez_compressed(
        ruta_archivo,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        metadatos=np.array([metadatos_json])  # Guardamos como array de 1 elemento
    )

    print(f"\n  {'='*55}")
    print(f"  MODELO GUARDADO EXITOSAMENTE")
    print(f"  {'='*55}")
    print(f"  Nombre   : {nombre_modelo}")
    print(f"  Archivo  : {nombre_archivo}")
    print(f"  Ruta     : {ruta_archivo}")
    print(f"  Tamaño W1: {W1.shape}")
    print(f"  Tamaño W2: {W2.shape}")
    if precision_test is not None:
        print(f"  Precisión: {precision_test:.2f}%")
    print(f"  {'='*55}")

    return ruta_archivo


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo(ruta_archivo=None, nombre_modelo=None):
    """
    Carga un modelo previamente guardado desde un archivo .npz.

    Se puede especificar:
      - ruta_archivo: ruta completa al archivo .npz
      - nombre_modelo: nombre parcial; se busca el archivo más reciente
                       que contenga ese nombre en la carpeta de modelos.

    Si no se especifica ninguno, se carga el modelo más reciente.

    Parámetros
    ──────────
    ruta_archivo : str, opcional
        Ruta completa al archivo .npz del modelo.
    nombre_modelo : str, opcional
        Nombre (parcial) del modelo para buscar en la carpeta de modelos.

    Retorna
    ───────
    W1, b1, W2, b2 : np.array
        Pesos y sesgos del modelo cargado.
    metadatos : dict
        Información del modelo (nombre, fecha, precisión, etc.).
    """
    if ruta_archivo is None:
        # Buscar en la carpeta de modelos
        if not os.path.exists(CARPETA_MODELOS):
            raise FileNotFoundError(
                f"No se encontró la carpeta de modelos: {CARPETA_MODELOS}\n"
                f"Primero entrena y guarda un modelo."
            )

        # Listar todos los archivos .npz
        archivos = [f for f in os.listdir(CARPETA_MODELOS) if f.endswith('.npz')]

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

    # Cargar el archivo .npz
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    datos = np.load(ruta_archivo, allow_pickle=True)

    W1 = datos['W1']
    b1 = datos['b1']
    W2 = datos['W2']
    b2 = datos['b2']

    # Recuperar metadatos
    metadatos_json = str(datos['metadatos'][0])
    metadatos = json.loads(metadatos_json)

    print(f"\n  {'='*55}")
    print(f"  MODELO CARGADO EXITOSAMENTE")
    print(f"  {'='*55}")
    print(f"  Nombre   : {metadatos.get('nombre_modelo', 'N/A')}")
    print(f"  Fecha    : {metadatos.get('fecha_guardado', 'N/A')}")
    print(f"  Tamaño W1: {W1.shape}")
    print(f"  Tamaño W2: {W2.shape}")
    prec = metadatos.get('precision_test')
    if prec is not None:
        print(f"  Precisión: {prec:.2f}%")
    print(f"  {'='*55}")

    return W1, b1, W2, b2, metadatos


# ─────────────────────────────────────────────────────────────────────────────
# PROBAR MODELO CON DATASET DE TEST
# ─────────────────────────────────────────────────────────────────────────────

def probar_modelo(W1, b1, W2, b2, X_test, y_test, nombre_modelo="Modelo"):
    """
    Evalúa un modelo cargado sobre un dataset de test completo.

    Realiza el forward pass sobre X_test y calcula:
      - Precisión global (% de aciertos)
      - Precisión por dígito (0-9)
      - Matriz de confusión simplificada

    Parámetros
    ──────────
    W1, b1, W2, b2 : np.array
        Pesos del modelo (cargado o recién entrenado).
    X_test : np.array, forma (N, 784)
        Imágenes de test aplanadas y normalizadas.
    y_test : np.array, forma (N,)
        Etiquetas numéricas reales (0-9).
    nombre_modelo : str
        Nombre del modelo para mostrar en los resultados.

    Retorna
    ───────
    acc_global : float
        Precisión global en porcentaje.
    acc_por_digito : np.array, forma (10,)
        Precisión para cada dígito (0-9).
    """
    # Importar forward y funciones necesarias
    from Fuctions import forward, precision

    # Forward pass sobre todo el test set
    _, _, _, A2 = forward(X_test, W1, b1, W2, b2)
    y_pred = np.argmax(A2, axis=1)

    # Precisión global
    acc_global = precision(y_pred, y_test)

    # Precisión por dígito
    acc_por_digito = np.zeros(10)
    conteo_por_digito = np.zeros(10, dtype=int)

    for digito in range(10):
        mascara = (y_test == digito)
        conteo_por_digito[digito] = np.sum(mascara)
        if conteo_por_digito[digito] > 0:
            acc_por_digito[digito] = precision(y_pred[mascara], y_test[mascara])

    # Mostrar resultados
    print(f"\n  {'='*60}")
    print(f"  RESULTADOS DE TEST — {nombre_modelo}")
    print(f"  {'='*60}")
    print(f"  Muestras evaluadas: {len(y_test)}")
    print(f"  Precisión global  : {acc_global:.2f}%")
    print(f"  {'─'*60}")
    print(f"  Precisión por dígito:")
    for digito in range(10):
        barra = '█' * int(acc_por_digito[digito] / 2)
        print(f"    Dígito {digito}: {acc_por_digito[digito]:5.1f}%  "
              f"({conteo_por_digito[digito]:4d} muestras)  {barra}")
    print(f"  {'='*60}")

    return acc_global, acc_por_digito

