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
from Worker import train_on_batch, worker_entrenar_particion

# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS DEL ALGORITMO MULTIPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

NUM_PARTICIONES = 4     # Número de subconjuntos en que dividimos los datos
EPOCAS = 100            # Número total de épocas (rondas de sincronización)
LEARNING_RATE = 0.1     # Tasa de aprendizaje
INTERVALO_LOG = 10      # Cada cuántas épocas imprimimos progreso
MAX_WORKERS = NUM_PARTICIONES # Máximo de procesos concurrentes

def particionar_dataset(X_train, Y_train, y_train, num_particiones):
    """
    Divide el dataset de entrenamiento en K particiones iguales.

    Parámetros
    ──────────
    X_train : np.array, forma (N, 784)
    Y_train : np.array, forma (N, 10)
    y_train : np.array, forma (N,)
    num_particiones : int

    Retorna
    ───────
    particiones : lista de K tuplas (X_k, Y_k, y_k)
    """
    N = X_train.shape[0]

    # Mezclar los índices aleatoriamente
    indices = np.random.permutation(N)

    # Aplicar la permutación
    X_mezclado = X_train[indices]
    Y_mezclado = Y_train[indices]
    y_mezclado = y_train[indices]

    # Dividir en K partes
    X_partes = np.array_split(X_mezclado, num_particiones)
    Y_partes = np.array_split(Y_mezclado, num_particiones)
    y_partes = np.array_split(y_mezclado, num_particiones)

    particiones = []
    for k in range(num_particiones):
        particiones.append((X_partes[k], Y_partes[k], y_partes[k]))

    print(f"\n  Dataset dividido en {num_particiones} particiones:")
    for k, (X_k, Y_k, y_k) in enumerate(particiones):
        digitos_unicos = np.unique(y_k)
        print(f"    Partición {k+1}: {X_k.shape[0]:5d} muestras  │  "
              f"Dígitos presentes: {digitos_unicos}")

    return particiones

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: PROMEDIAR LOS PESOS DE TODAS LAS PARTICIONES
# ─────────────────────────────────────────────────────────────────────────────

def promediar_pesos(lista_pesos):
    """
    Promedia los pesos de K redes tras entrenar cada una con su partición.
    """
    K = len(lista_pesos)

    lista_W1 = [pesos[0] for pesos in lista_pesos]
    lista_b1 = [pesos[1] for pesos in lista_pesos]
    lista_W2 = [pesos[2] for pesos in lista_pesos]
    lista_b2 = [pesos[3] for pesos in lista_pesos]

    W1_prom = np.mean(np.array(lista_W1), axis=0)    # (784, 128)
    b1_prom = np.mean(np.array(lista_b1), axis=0)    # (1, 128)
    W2_prom = np.mean(np.array(lista_W2), axis=0)    # (128, 10)
    b2_prom = np.mean(np.array(lista_b2), axis=0)    # (1, 10)

    return W1_prom, b1_prom, W2_prom, b2_prom


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: ALGORITMO COMPLETO CON MULTIPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_multiprocessing(X_train, Y_train, y_train, X_test, y_test,
                             num_particiones, epocas, lr, intervalo_log, max_workers):
    """
    Ejecuta el Algoritmo Multiprocessing completo con entrenamiento paralelo.
    
    Los pesos se promedian al final de cada época y ese promedio se convierte
    en el punto de partida de la siguiente época.
    """

    print("\n" + "=" * 70)
    print("  ALGORITMO MULTIPROCESSING — ENTRENAMIENTO FEDERADO PARALELO")
    print("=" * 70)
    print(f"  Particiones       : {num_particiones}")
    print(f"  Épocas totales    : {epocas}")
    print(f"  Learning Rate     : {lr}")
    print(f"  Núcleos disponibles: {cpu_count()}")
    print(f"  Máx. procesos     : {max_workers}")
    print(f"  Muestras totales  : {X_train.shape[0]}")
    print(f"  Muestras por red  : ~{X_train.shape[0] // num_particiones}")

    # ── PASO 1: Inicializar pesos globales una sola vez ───────────────────
    print("\n" + "=" * 70)
    print("  PASO 1: INICIALIZANDO PESOS GLOBALES")
    print("=" * 70)
    W1, b1, W2, b2 = inicializar_pesos()

    # ── PASO 2: Particionar el dataset (particiones FIJAS) ─────────────────
    print("\n" + "=" * 70)
    print("  PASO 2: PARTICIONANDO EL DATASET")
    print("=" * 70)
    particiones = particionar_dataset(X_train, Y_train, y_train, num_particiones)

    # ── PASO 3: Bucle principal — entrenamiento federado paralelo ─────────
    print("\n" + "=" * 70)
    print("  PASO 3: ENTRENAMIENTO FEDERADO PARALELO (PROMEDIADO POR ÉPOCA)")
    print("=" * 70)

    historial_loss = []
    historial_acc  = []
    historial_acc_test = []

    # Historiales por partición
    hist_loss_parts = [[] for _ in range(num_particiones)]
    hist_acc_parts  = [[] for _ in range(num_particiones)]

    for epoca in range(1, epocas + 1):
        
        tiempo_inicio_epoca = time.time()

        # ── a) Guardar estado global actual ───────────────────────────────
        W1_global = np.copy(W1)
        b1_global = np.copy(b1)
        W2_global = np.copy(W2)
        b2_global = np.copy(b2)

        # ── b) Preparar argumentos para los workers ──────────────────────
        # Cada worker recibe: (id, X_k, Y_k, y_k, W1_global, b1_global, W2_global, b2_global, lr)
        args_lista = []
        for idx, (X_k, Y_k, y_k) in enumerate(particiones):
            args = (idx, X_k, Y_k, y_k, W1_global, b1_global, W2_global, b2_global, lr)
            args_lista.append(args)

        # ── c) Entrenar TODAS las particiones EN PARALELO ──────────────────
        # multiprocessing.Pool.map() distribuye las tareas entre los workers
        # Máximo de procesos concurrentes = max_workers
        try:
            with Pool(processes=max_workers) as pool:
                resultados = pool.map(worker_entrenar_particion, args_lista)
        except Exception as e:
            print(f"\n  ✗ Error durante multiprocessing en época {epoca}: {e}")
            return 1

        # ── d) Procesar resultados y actualizar pesos ────────────────────
        lista_pesos_epoca = []
        
        for resultado in resultados:
            idx, W1_k, b1_k, W2_k, b2_k, loss_k, acc_k = resultado
            lista_pesos_epoca.append((W1_k, b1_k, W2_k, b2_k))
            hist_loss_parts[idx].append(loss_k)
            hist_acc_parts[idx].append(acc_k)

        # ── e) Promediar los pesos de todas las particiones ──────────────
        W1, b1, W2, b2 = promediar_pesos(lista_pesos_epoca)

        # ── f) Evaluación global ──────────────────────────────────────────
        # Forward pass con todos los datos de entrenamiento
        Z1_all, A1_all, Z2_all, A2_all = forward(X_train, W1, b1, W2, b2)

        # Pérdida global
        loss = cross_entropy(A2_all, Y_train)
        historial_loss.append(loss)

        # Precisión global en entrenamiento
        y_pred_train = np.argmax(A2_all, axis=1)
        acc_train = precision(y_pred_train, y_train)
        historial_acc.append(acc_train)

        # Precisión en test
        y_pred_test = predecir(X_test, W1, b1, W2, b2)
        acc_test = precision(y_pred_test, y_test)
        historial_acc_test.append(acc_test)

        # ── g) Log de progreso ────────────────────────────────────────────
        tiempo_epoca = time.time() - tiempo_inicio_epoca
        
        if epoca % intervalo_log == 0 or epoca == 1:
            print(f"\n  {'─'*68}")
            print(f"  ÉPOCA {epoca}/{epocas} — RESULTADOS TRAS PROMEDIADO")
            print(f"  {'─'*68}")

            # Mostrar resultados de particiones individuales
            for idx in range(num_particiones):
                l_k = hist_loss_parts[idx][-1]
                a_k = hist_acc_parts[idx][-1]
                print(f"    [Proc {idx+1}] Partición {idx+1:2d}: Loss={l_k:.4f}  Acc={a_k:.1f}%")

            # Mostrar resultado del modelo promediado
            print(f"  {'─'*68}")
            print(f"    ✓ PROMEDIADO → Loss: {loss:.4f} │ "
                  f"Acc Train: {acc_train:.1f}% │ "
                  f"Acc Test: {acc_test:.1f}% │ "
                  f"Tiempo: {tiempo_epoca:.2f}s")

    # ── PASO 4: Evaluación final ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUACIÓN FINAL")
    print("=" * 70)
    y_pred_test = predecir(X_test, W1, b1, W2, b2)
    acc_final = precision(y_pred_test, y_test)
    print(f"\n  ✓ Precisión FINAL del modelo Multiprocessing en TEST: {acc_final:.2f}%")

    # ── Graficar resultados ─────────────────────────────────────────────
    graficar_diego(historial_loss, historial_acc, historial_acc_test,
                   hist_loss_parts, hist_acc_parts, num_particiones)

    # Guardar el modelo entrenado
    guardar_modelo(
        W1, b1, W2, b2,
        nombre_modelo='MultiProcessingNN',
        precision_test=acc_final,
        epocas=epocas,
        learning_rate=LEARNING_RATE,
        info_extra={
            'num_particiones': num_particiones,
            'max_workers': max_workers,
            'nucleos_disponibles': cpu_count()
        }
    )
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  ALGORITMO MULTIPROCESSING — RED NEURONAL FEDERADA — MNIST")
    print("  Sin librerías de ML, solo NumPy + multiprocessing")
    print("=" * 70)

    # ── 1. Cargar el dataset ─────────────────────────────────────────────────
    X_all, y_all = cargar_mnist()

    # ── 2. Preprocesar: aplanar, normalizar, one-hot, split 70/30 ────────────
    X_train, Y_train, y_train, X_test, Y_test, y_test = preprocesar(X_all, y_all)

    # ── 3. Ejecutar el Algoritmo Multiprocessing ──────────────────────────────
    entrenar_multiprocessing(
        X_train, Y_train, y_train,
        X_test, y_test,
        num_particiones=NUM_PARTICIONES,
        epocas=EPOCAS,
        lr=LEARNING_RATE,
        intervalo_log=INTERVALO_LOG,
        max_workers=MAX_WORKERS
    )