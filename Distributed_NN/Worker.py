import sys
import os
import numpy as np
import pickle
import socket
from multiprocessing import Pool, cpu_count
import time

# ── Agregar el directorio padre al path para acceder al paquete Utils ─────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.DatasetHandling import cargar_mnist, preprocesar
from Utils.Graphics import graficar_resultados, graficar_diego
from Utils.Fuctions import forward, backward, cross_entropy, precision, predecir
from Utils.WeightsHandling import inicializar_pesos, actualizar_pesos
from Utils.ModelPersistence import guardar_modelo, cargar_modelo

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN WORKER PARA MULTIPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def worker_entrenar_particion(args):
    """
    Función worker para entrenar una partición en un proceso separado.
    
    Se ejecuta en un proceso hijo que entrena una partición con los pesos
    globales recibidos y devuelve los pesos actualizados.
    
    Parámetros
    ──────────
    args : tupla
        (id_particion, X_k, Y_k, y_k, W1_global, b1_global, W2_global, b2_global, lr)
    
    Retorna
    ───────
    tupla (id_particion, W1_actualizado, b1_actualizado, W2_actualizado, b2_actualizado,
           loss_particion, acc_particion)
    """
    (idx_part, X_k, Y_k, y_k, W1_glob, b1_glob, W2_glob, b2_glob, lr) = args
    
    # Copiar pesos para no mutar los originales (en el proceso hijo)
    W1_local = np.copy(W1_glob)
    b1_local = np.copy(b1_glob)
    W2_local = np.copy(W2_glob)
    b2_local = np.copy(b2_glob)
    
    # Forward Pass
    Z1, A1, Z2, A2 = forward(X_k, W1_local, b1_local, W2_local, b2_local)
    
    # Backward Pass
    dW1, db1, dW2, db2 = backward(X_k, Y_k, Z1, A1, A2, W2_local)
    
    # Actualizar Pesos
    W1_local, b1_local, W2_local, b2_local = actualizar_pesos(
        W1_local, b1_local, W2_local, b2_local,
        dW1, db1, dW2, db2, lr
    )
    
    # Calcular métricas de la partición
    loss_k = cross_entropy(A2, Y_k)
    acc_k = precision(np.argmax(A2, axis=1), y_k)
    
    return (idx_part, W1_local, b1_local, W2_local, b2_local, loss_k, acc_k)

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: ENTRENAR UNA PARTICIÓN POR UN SOLO PASO (train_on_batch)
# ─────────────────────────────────────────────────────────────────────────────

def train_on_batch(X_k, Y_k, W1, b1, W2, b2, lr):
    """
    Entrena la red con UN SOLO batch de datos (una partición) y retorna
    los pesos actualizados.

    Parámetros
    ──────────
    X_k  : np.array, forma (N_k, 784) — datos de la partición k
    Y_k  : np.array, forma (N_k, 10)  — etiquetas one-hot de la partición k
    W1, b1, W2, b2 : pesos de la red (se COPIAN internamente)
    lr   : float — learning rate

    Retorna
    ───────
    W1, b1, W2, b2 : pesos actualizados después de una pasada
    """
    # ── Copiar pesos para no mutar los originales ────────────────────────────
    # Cada partición trabaja con su propia copia independiente
    W1_local = np.copy(W1)
    b1_local = np.copy(b1)
    W2_local = np.copy(W2)
    b2_local = np.copy(b2)

    # ── Forward Pass ─────────────────────────────────────────────────────────
    Z1, A1, Z2, A2 = forward(X_k, W1_local, b1_local, W2_local, b2_local)

    # ── Backward Pass ────────────────────────────────────────────────────────
    dW1, db1, dW2, db2 = backward(X_k, Y_k, Z1, A1, A2, W2_local)

    # ── Actualizar Pesos ─────────────────────────────────────────────────────
    W1_local, b1_local, W2_local, b2_local = actualizar_pesos(
        W1_local, b1_local, W2_local, b2_local,
        dW1, db1, dW2, db2, lr
    )

    return W1_local, b1_local, W2_local, b2_local