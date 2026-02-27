"""
=============================================================================
  ALGORITMO DE DIEGO — ENTRENAMIENTO FEDERADO CON PROMEDIADO POR ÉPOCA
=============================================================================

  Diego:
    Época 1: entrenar + promediar → nuevo punto de partida
    Época 2: entrenar + promediar → nuevo punto de partida
    ...
    Época 100: entrenar + promediar → modelo final

  En Diego, los pesos se "sincronizan" después de CADA época.
  Esto es más cercano al verdadero Federated Averaging (FedAvg).

FLUJO DEL ALGORITMO POR ÉPOCA
──────────────────────────────
  Para cada época e = 1, 2, ..., E:

  ┌─────────────────────────────────────────────────────────────┐
  │  a) Copiar el estado global actual (W1, b1, W2, b2)        │
  │                                                             │
  │  b) Para cada partición k = 1, ..., K:                      │
  │     - Recibir COPIA de los pesos globales                   │
  │     - Hacer UN forward pass con su batch de datos           │
  │     - Calcular gradientes (backward pass)                   │
  │     - Aplicar UNA actualización de pesos                    │
  │     - Devolver los pesos actualizados                       │
  │                                                             │
  │  c) Promediar los K conjuntos de pesos actualizados         │
  │     W_global = (1/K) · Σ W_k                                │
  │                                                             │
  │  d) Evaluar el modelo promediado sobre TODOS los datos      │
  │     para registrar loss y accuracy globales                 │
  └─────────────────────────────────────────────────────────────┘

ARQUITECTURA (IDÉNTICA A BasicNeuralNetwork.py)
────────────────────────────────────────────────
  Capa de Entrada  : 784 neuronas  (28×28 píxeles aplanados)
        ↓
  Capa Oculta      : 128 neuronas  (ReLU)
        ↓
  Capa de Salida   : 10 neuronas   (Softmax)

=============================================================================
"""

import sys
import os
import numpy as np

# ── Agregar el directorio de módulos al path para encontrar los módulos ─────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Modules'))

from DatasetHandling import cargar_mnist, preprocesar
from Graphics import graficar_resultados, graficar_diego
from Fuctions import forward, backward, cross_entropy, precision, predecir
from WeightsHandling import inicializar_pesos, actualizar_pesos
from ModelPersistence import guardar_modelo, cargar_modelo


# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS DEL ALGORITMO DE DIEGO
# ─────────────────────────────────────────────────────────────────────────────

NUM_PARTICIONES = 5     # Número de subconjuntos en que dividimos los datos
EPOCAS = 100            # Número total de épocas (rondas de sincronización)
LEARNING_RATE = 0.1     # Tasa de aprendizaje
INTERVALO_LOG = 10      # Cada cuántas épocas imprimimos progreso


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: PARTICIONAR EL DATASET
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: PROMEDIAR LOS PESOS DE TODAS LAS PARTICIONES
# ─────────────────────────────────────────────────────────────────────────────

def promediar_pesos(lista_pesos):
    """
    Promedia los pesos de K redes tras entrenar cada una con su partición.

    ─────────────────────
    Cada partición propone "moverse" en una dirección diferente del espacio
    de parámetros. El promediado encuentra el punto central de todas esas
    propuestas. Es como pedir direcciones a K personas y caminar hacia
    el promedio de lo que te dijeron.
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
# PASO 4: ALGORITMO COMPLETO DE DIEGO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_diego(X_train, Y_train, y_train, X_test, y_test,
                   num_particiones,epocas,lr,intervalo_log):
    """
    Ejecuta el Algoritmo de Diego completo.
    """

    print("\n" + "=" * 60)
    print("  ALGORITMO DE DIEGO — ENTRENAMIENTO FEDERADO POR ÉPOCA")
    print("=" * 60)
    print(f"  Particiones       : {num_particiones}")
    print(f"  Épocas totales    : {epocas}")
    print(f"  Learning Rate     : {lr}")
    print(f"  Muestras totales  : {X_train.shape[0]}")
    print(f"  Muestras por red  : ~{X_train.shape[0] // num_particiones}")

    # ── PASO 1: Inicializar pesos UNA sola vez ───────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 1: INICIALIZANDO PESOS GLOBALES")
    print("=" * 60)
    W1, b1, W2, b2 = inicializar_pesos()

    # ── PASO 2: Particionar el dataset (particiones FIJAS) ────────────────────
    print("\n" + "=" * 60)
    print("  PASO 2: PARTICIONANDO EL DATASET")
    print("=" * 60)
    particiones = particionar_dataset(X_train, Y_train, y_train, num_particiones)

    # ── PASO 3: Bucle principal — una ronda de federación por época ──────────
    print("\n" + "=" * 60)
    print("  PASO 3: ENTRENAMIENTO FEDERADO (PROMEDIADO POR ÉPOCA)")
    print("=" * 60)

    historial_loss = []
    historial_acc  = []
    historial_acc_test = []

    # Historiales por partición: para cada partición k, guardamos su loss y acc
    # en cada época ANTES del promediado (lo que cada red individual aprendió)
    hist_loss_parts = [[] for _ in range(num_particiones)]
    hist_acc_parts  = [[] for _ in range(num_particiones)]

    for epoca in range(1, epocas + 1):

        # ── a) Estado global actual ──────────────────────────────────────────
        # Guardamos una referencia a los pesos actuales. No necesitamos
        # copiar aquí porque train_on_batch() ya copia internamente.
        # Los pesos W1, b1, W2, b2 NO se modifican en este punto.
        W1_global = W1
        b1_global = b1
        W2_global = W2
        b2_global = b2

        # ── b) Entrenar cada partición de forma independiente ────────────────
        # Cada partición recibe los MISMOS pesos globales y hace UNA pasada
        #
        # Visualmente para K=3:
        #
        #   Pesos globales ──┬── Partición 1 ── train_on_batch() ── pesos_1
        #                    ├── Partición 2 ── train_on_batch() ── pesos_2
        #                    └── Partición 3 ── train_on_batch() ── pesos_3
        #
        #   pesos_1, pesos_2, pesos_3 ──── promediar() ──── nuevos pesos globales

        lista_pesos_epoca = []

        for k, (X_k, Y_k, y_k) in enumerate(particiones):
            # train_on_batch copia los pesos globales, hace forward + backward
            # + actualización, y devuelve los pesos modificados
            W1_k, b1_k, W2_k, b2_k = train_on_batch(
                X_k, Y_k,
                W1_global, b1_global, W2_global, b2_global,
                lr
            )
            lista_pesos_epoca.append((W1_k, b1_k, W2_k, b2_k))

            # ── Métricas por partición (ANTES del promediado) ────────────────
            # Evaluamos qué tan bien le fue a ESTA partición con SUS datos
            _, _, _, A2_k = forward(X_k, W1_k, b1_k, W2_k, b2_k)
            loss_k = cross_entropy(A2_k, Y_k)
            acc_k  = precision(np.argmax(A2_k, axis=1), y_k)
            hist_loss_parts[k].append(loss_k)
            hist_acc_parts[k].append(acc_k)

        # ── c) Promediar los pesos de todas las particiones ──────────────────
        # Este promediado es la "sincronización" que hace especial a Diego.
        # Después de esto, todos los pesos parciales se funden en uno solo.
        #
        #   W1_nuevo[i,j] = (1/K) · Σ_{k=1}^{K} W1_k[i,j]
        #
        # Los nuevos pesos globales incorporan lo aprendido por TODAS las
        # particiones en esta época.
        W1, b1, W2, b2 = promediar_pesos(lista_pesos_epoca)

        # ── d) Evaluación global ─────────────────────────────────────────────
        # Evaluamos el modelo promediado sobre TODOS los datos de entrenamiento
        # (no solo una partición) para tener métricas globales consistentes.
        #
        # Forward pass con todos los datos:
        #   Z1 = X_train · W1 + b1    (N, 784) @ (784, 128) = (N, 128)
        #   A1 = ReLU(Z1)             (N, 128)
        #   Z2 = A1 · W2 + b2         (N, 128) @ (128, 10) = (N, 10)
        #   A2 = Softmax(Z2)          (N, 10)
        Z1_all, A1_all, Z2_all, A2_all = forward(X_train, W1, b1, W2, b2)

        # Pérdida global (cross-entropy sobre TODOS los datos de entrenamiento)
        loss = cross_entropy(A2_all, Y_train)
        historial_loss.append(loss)

        # Precisión global sobre entrenamiento
        y_pred_train = np.argmax(A2_all, axis=1)
        acc_train = precision(y_pred_train, y_train)
        historial_acc.append(acc_train)

        # Precisión en test (registrar CADA época para la gráfica)
        y_pred_test = predecir(X_test, W1, b1, W2, b2)
        acc_test = precision(y_pred_test, y_test)
        historial_acc_test.append(acc_test)

        # ── Log de progreso ───────────────────────────────────────────────────
        # Mostramos el detalle por partición + el resultado del promediado
        if epoca % intervalo_log == 0 or epoca == 1:
            print(f"\n  {'─'*58}")
            print(f"  ÉPOCA {epoca}/{epocas} — RESULTADOS TRAS PROMEDIADO")
            print(f"  {'─'*58}")

            # Mostrar qué hizo cada partición individualmente
            for k in range(num_particiones):
                l_k = hist_loss_parts[k][-1]
                a_k = hist_acc_parts[k][-1]
                print(f"    Partición {k+1}: Loss={l_k:.4f}  Acc={a_k:.1f}%")

            # Mostrar el resultado del modelo promediado
            print(f"  {'─'*58}")
            print(f"    PROMEDIADO → Loss: {loss:.4f} │ "
                  f"Acc Train: {acc_train:.1f}% │ "
                  f"Acc Test: {acc_test:.1f}%")

    # ── PASO 4: Evaluación final ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUACIÓN FINAL")
    print("=" * 60)
    y_pred_test = predecir(X_test, W1, b1, W2, b2)
    acc_final = precision(y_pred_test, y_test)
    print(f"\n  Precisión FINAL del modelo Diego en TEST: {acc_final:.2f}%")


    # ── Graficar resultados ────────────────────────────────────────────────
    graficar_diego(historial_loss, historial_acc, historial_acc_test,
                   hist_loss_parts, hist_acc_parts, NUM_PARTICIONES)

    # Guardar el modelo entrenado
    y_pred_final = predecir(X_test, W1, b1, W2, b2)
    acc_final = precision(y_pred_final, y_test)
    guardar_modelo(
        W1, b1, W2, b2,
        nombre_modelo='DiegoNN',
        precision_test=acc_final,
        epocas=EPOCAS,
        learning_rate=LEARNING_RATE,
        info_extra={'num_particiones': NUM_PARTICIONES}
    )

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  ALGORITMO DE DIEGO — RED NEURONAL FEDERADA — MNIST")
    print("  Sin librerías de ML, solo NumPy")
    print("=" * 60)

    # ── 1. Cargar el dataset ─────────────────────────────────────────────────
    X_all, y_all = cargar_mnist()

    # ── 2. Preprocesar: aplanar, normalizar, one-hot, split 70/30 ────────────
    X_train, Y_train, y_train, X_test, Y_test, y_test = preprocesar(X_all, y_all)

    # ── 3. Ejecutar el Algoritmo de Diego ─────────────────────────────────────
    entrenar_diego(
        X_train, Y_train, y_train,
        X_test, y_test,
        num_particiones=NUM_PARTICIONES,
        epocas=EPOCAS,
        lr=LEARNING_RATE,
        intervalo_log=INTERVALO_LOG
    )


