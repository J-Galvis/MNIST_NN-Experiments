"""
=============================================================================
  ALGORITMO DE ARNOVI — ENTRENAMIENTO DISTRIBUIDO CON PROMEDIADO DE PESOS
=============================================================================

ARQUITECTURA (IDÉNTICA A BasicNeuralNetwork.py)
────────────────────────────────────────────────
  Capa de Entrada  : 784 neuronas  (28×28 píxeles aplanados)
        ↓
  Capa Oculta      : 128 neuronas  (ReLU)
        ↓
  Capa de Salida   : 10 neuronas   (Softmax)

FLUJO DEL ALGORITMO DE ARNOVI
──────────────────────────────
  ┌─────────────────────────────────────────────────────────┐
  │  1. Cargar y preprocesar MNIST (70000 imágenes)         │
  │  2. Separar en train (70%) y test (30%)                 │
  │  3. Inicializar pesos UNA sola vez (W1, b1, W2, b2)     │
  │  4. Dividir train en K particiones iguales              │
  │  5. Para cada partición k = 1, 2, ..., K:               │
  │     a) Copiar los pesos iniciales                       │
  │     b) Entrenar con ese subconjunto por E épocas        │
  │     c) Guardar los pesos entrenados                     │
  │  6. Promediar los pesos de las K redes                  │
  │  7. Evaluar el modelo promediado en el set de test      │
  └─────────────────────────────────────────────────────────┘

=============================================================================
"""

import sys
import os
import numpy as np
import time

# ── Agregar el directorio padre al path para acceder al paquete Utils ─────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.DatasetHandling import cargar_mnist, preprocesar
from Utils.Graphics import graficar_arnovi
from Utils.Fuctions import forward, backward, cross_entropy, precision, predecir
from Utils.WeightsHandling import inicializar_pesos, actualizar_pesos
from Utils.ModelPersistence import guardar_modelo, cargar_modelo


# ─────────────────────────────────────────────────────────────────────────────
# HIPERPARÁMETROS DEL ALGORITMO DE ARNOVI
# ─────────────────────────────────────────────────────────────────────────────

NUM_PARTICIONES = 3     # Número de subconjuntos en que dividimos los datos
EPOCAS_POR_PARTICION = 100  # Cuántas épocas entrena cada mini-red
LEARNING_RATE = 0.3     # Tasa de aprendizaje (misma que BasicNeuralNetwork)
INTERVALO_LOG = 10      # Cada cuántas épocas imprimimos el progreso


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: PARTICIONAR EL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def particionar_dataset(X_train, Y_train, y_train, num_particiones):
    """
    Divide el dataset de entrenamiento en K particiones iguales.

    Parámetros
    ──────────
    X_train : np.array, forma (N, 784)
        Imágenes aplanadas y normalizadas (valores entre 0 y 1).
    Y_train : np.array, forma (N, 10)
        Etiquetas en formato one-hot.
    y_train : np.array, forma (N,)
        Etiquetas numéricas (0-9), usadas para medir precisión.
    num_particiones : int
        Número K de subconjuntos a crear.

    Retorna
    ───────
    particiones : lista de K tuplas, cada una con (X_k, Y_k, y_k)
    """
    N = X_train.shape[0]

    # Mezclar los índices aleatoriamente
    indices = np.random.permutation(N)

    # Aplicar la permutación a los datos
    # Esto reordena las filas de X_train, Y_train y y_train de la misma manera
    X_mezclado = X_train[indices]    # (N, 784) → reordenar filas
    Y_mezclado = Y_train[indices]    # (N, 10)  → mismo orden de mezcla
    y_mezclado = y_train[indices]    # (N,)     → mismo orden de mezcla

    # Dividir en K partes iguales (o casi iguales)
    # np.array_split retorna una lista de K arrays
    X_partes = np.array_split(X_mezclado, num_particiones)
    Y_partes = np.array_split(Y_mezclado, num_particiones)
    y_partes = np.array_split(y_mezclado, num_particiones)

    # Empaquetamos cada partición como una tupla (X_k, Y_k, y_k)
    particiones = []
    for k in range(num_particiones):
        particiones.append((X_partes[k], Y_partes[k], y_partes[k]))

    # ── Verificación: imprimir cuántas muestras tiene cada partición ─────────
    print(f"\n  Dataset dividido en {num_particiones} particiones:")
    for k, (X_k, Y_k, y_k) in enumerate(particiones):
        digitos_unicos = np.unique(y_k)
        print(f"    Partición {k+1}: {X_k.shape[0]:5d} muestras  │  "
              f"Dígitos presentes: {digitos_unicos}")

    return particiones


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: ENTRENAR UNA RED EN UNA PARTICIÓN
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_particion(X_k, Y_k, y_k, W1_init, b1_init, W2_init, b2_init,
                       id_particion, epocas, lr, intervalo_log,
                       X_test=None, y_test=None):
    """
    Entrena una red neuronal usando SOLO los datos de una partición.

    Parámetros
    ──────────
    X_k, Y_k, y_k : datos de la partición k
    W1_init, b1_init, W2_init, b2_init : pesos INICIALES (se copian)
    id_particion : int, número de la partición (para los logs)
    epocas : int, número de épocas de entrenamiento
    lr : float, learning rate
    intervalo_log : int, cada cuántas épocas se imprime progreso
    X_test, y_test : datos de test (opcionales, solo para log de progreso)

    Retorna
    ───────
    W1, b1, W2, b2 : pesos entrenados de esta partición
    historial_loss : lista con la pérdida en cada época
    historial_acc  : lista con la precisión en cada época
    """

    # ── Copiar los pesos iniciales para esta partición ────────────────────────
    # np.copy() crea una nueva matriz en memoria con los mismos valores
    # Esto garantiza que cada partición trabaja con su propia copia
    W1 = np.copy(W1_init)      # (784, 128) — copia independiente
    b1 = np.copy(b1_init)      # (1, 128)
    W2 = np.copy(W2_init)      # (128, 10)
    b2 = np.copy(b2_init)      # (1, 10)

    historial_loss = []
    historial_acc  = []

    N_k = X_k.shape[0]     # Número de muestras en esta partición

    print(f"\n  {'─'*55}")
    print(f"  PARTICIÓN {id_particion} — {N_k} muestras")
    print(f"  {'─'*55}")

    for epoca in range(1, epocas + 1):

        # ── Forward Pass ─────────────────────────────────────────────────────
        
        # Exactamente igual que en BasicNeuralNetwork, pero con X_k (menos datos)
        #   Z1 = X_k · W1 + b1     →  (N_k, 784) @ (784, 128) = (N_k, 128)
        #   A1 = ReLU(Z1)          →  (N_k, 128)
        #   Z2 = A1 · W2 + b2      →  (N_k, 128) @ (128, 10) = (N_k, 10)
        #   A2 = Softmax(Z2)       →  (N_k, 10)
        Z1, A1, Z2, A2 = forward(X_k, W1, b1, W2, b2)

        # ── Pérdida ──────────────────────────────────────────────────────────
        loss = cross_entropy(A2, Y_k)
        historial_loss.append(loss)

        # ── Precisión en esta partición ───────────────────────────────────────
        y_pred_k = np.argmax(A2, axis=1)
        acc_k = precision(y_pred_k, y_k)
        historial_acc.append(acc_k)

        # ── Backward Pass ────────────────────────────────────────────────────
        # Los gradientes se calculan con N_k muestras en lugar de N
        # dW1 = (1/N_k) · X_kᵀ · dZ1   ← nota: se divide por N_k, no por N
        dW1, db1, dW2, db2 = backward(X_k, Y_k, Z1, A1, A2, W2)

        # ── Actualizar pesos ─────────────────────────────────────────────────
        W1, b1, W2, b2 = actualizar_pesos(W1, b1, W2, b2,
                                           dW1, db1, dW2, db2, lr)

        # ── Log de progreso ──────────────────────────────────────────────────
        if epoca % intervalo_log == 0 or epoca == 1:
            log_msg = (f"    [P{id_particion}] Época {epoca:4d}/{epocas} │ "
                       f"Loss: {loss:.4f} │ Acc Part: {acc_k:.1f}%")

            # Si tenemos datos de test, mostramos también la precisión en test
            if X_test is not None and y_test is not None:
                y_pred_test = predecir(X_test, W1, b1, W2, b2)
                acc_test = precision(y_pred_test, y_test)
                log_msg += f" │ Acc Test: {acc_test:.1f}%"

            print(log_msg)

    return W1, b1, W2, b2, historial_loss, historial_acc


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: PROMEDIAR LOS PESOS DE TODAS LAS PARTICIONES
# ─────────────────────────────────────────────────────────────────────────────

def promediar_pesos(lista_pesos):
    """
    Promedia los pesos de K redes entrenadas independientemente.

    Parámetros
    ──────────
    lista_pesos : lista de K tuplas, cada una con (W1_k, b1_k, W2_k, b2_k)

    Retorna
    ───────
    W1_prom, b1_prom, W2_prom, b2_prom : pesos promediados
    """
    K = len(lista_pesos)

    # Separamos cada parámetro en su propia lista
    # lista_W1 contendrá: [W1_red1, W1_red2, ..., W1_redK]
    # Cada W1_k tiene forma (784, 128)
    lista_W1 = [pesos[0] for pesos in lista_pesos]   # K matrices de (784, 128)
    lista_b1 = [pesos[1] for pesos in lista_pesos]   # K matrices de (1, 128)
    lista_W2 = [pesos[2] for pesos in lista_pesos]   # K matrices de (128, 10)
    lista_b2 = [pesos[3] for pesos in lista_pesos]   # K matrices de (1, 10)

    # np.array() apila las matrices: lista de K matrices (784,128) → (K, 784, 128)
    # np.mean(axis=0) promedia sobre el eje K:  (K, 784, 128) → (784, 128)
    W1_prom = np.mean(np.array(lista_W1), axis=0)    # (784, 128)
    b1_prom = np.mean(np.array(lista_b1), axis=0)    # (1, 128)
    W2_prom = np.mean(np.array(lista_W2), axis=0)    # (128, 10)
    b2_prom = np.mean(np.array(lista_b2), axis=0)    # (1, 10)

    print(f"\n  Pesos promediados de {K} redes:")
    print(f"    W1_prom: {W1_prom.shape}  (pesos entrada → oculta)")
    print(f"    b1_prom: {b1_prom.shape}  (sesgos capa oculta)")
    print(f"    W2_prom: {W2_prom.shape}  (pesos oculta → salida)")
    print(f"    b2_prom: {b2_prom.shape}  (sesgos capa salida)")

    return W1_prom, b1_prom, W2_prom, b2_prom


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: ALGORITMO COMPLETO DE ARNOVI
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_arnovi(X_train, Y_train, y_train, X_test, y_test,
                    num_particiones,epocas,lr,intervalo_log):
    """
    Ejecuta el Algoritmo de Arnovi completo:
      1. Inicializar pesos (una sola vez)
      2. Particionar el dataset de entrenamiento
      3. Entrenar una red por partición (todas desde los mismos pesos)
      4. Promediar los pesos de todas las redes
      5. Evaluar el modelo promediado

    COMPARACIÓN CON BasicNeuralNetwork.py
    ──────────────────────────────────────
    BasicNN:  UNA red entrena con TODOS los 49000 datos por 100 épocas
    Arnovi:   K redes entrenan con 49000/K datos cada una por 100 épocas
              y luego se promedian los K conjuntos de pesos

    Cada red individual ve MENOS datos, pero el promediado aprovecha
    la "sabiduría colectiva" de todas las redes.
    """
    tiempo_inicio = time.time()

    print("\n" + "=" * 60)
    print("  ALGORITMO DE ARNOVI — ENTRENAMIENTO DISTRIBUIDO")
    print("=" * 60)
    print(f"  Particiones       : {num_particiones}")
    print(f"  Épocas por red    : {epocas}")
    print(f"  Learning Rate     : {lr}")
    print(f"  Muestras totales  : {X_train.shape[0]}")
    print(f"  Muestras por red  : ~{X_train.shape[0] // num_particiones}")

    # ── PASO 1: Inicializar pesos UNA sola vez ───────────────────────────────
    # Todas las mini-redes partirán de estos mismos pesos.
    # Esto es clave: si cada red empezara de pesos aleatorios diferentes,
    # podrían terminar en regiones muy distintas del espacio de parámetros,
    # y el promediado no tendría sentido geométrico.
    print("\n" + "=" * 60)
    print("  PASO 1: INICIALIZANDO PESOS COMPARTIDOS")
    print("=" * 60)
    W1_init, b1_init, W2_init, b2_init = inicializar_pesos()

    # ── PASO 2: Dividir el dataset en particiones ────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 2: PARTICIONANDO EL DATASET")
    print("=" * 60)
    particiones = particionar_dataset(X_train, Y_train, y_train, num_particiones)

    # ── PASO 3: Entrenar una red por partición ───────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 3: ENTRENANDO UNA RED POR PARTICIÓN")
    print("=" * 60)

    # Aquí guardaremos los pesos finales de cada red
    lista_pesos_entrenados = []

    # También guardamos el historial completo para graficar
    historiales_loss = []   # Lista de K listas de pérdida
    historiales_acc  = []   # Lista de K listas de precisión

    for k, (X_k, Y_k, y_k) in enumerate(particiones):

        # Entrenar la red k con su partición
        # Nota: pasamos los pesos INICIALES (W1_init, etc.)
        # La función entrenar_particion() los COPIA internamente
        W1_k, b1_k, W2_k, b2_k, hist_loss_k, hist_acc_k = entrenar_particion(
            X_k, Y_k, y_k,
            W1_init, b1_init, W2_init, b2_init,
            id_particion=k + 1,
            epocas=epocas,
            lr=lr,
            intervalo_log=intervalo_log,
            X_test=X_test,
            y_test=y_test
        )

        # Guardar los pesos entrenados de esta partición
        lista_pesos_entrenados.append((W1_k, b1_k, W2_k, b2_k))
        historiales_loss.append(hist_loss_k)
        historiales_acc.append(hist_acc_k)

        # Evaluar esta red individual en test
        y_pred_k = predecir(X_test, W1_k, b1_k, W2_k, b2_k)
        acc_k = precision(y_pred_k, y_test)
        print(f"  → Partición {k+1} terminada │ Acc Test individual: {acc_k:.2f}%")

    # ── PASO 4: Promediar los pesos ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 4: PROMEDIANDO PESOS DE TODAS LAS REDES")
    print("=" * 60)
    W1_final, b1_final, W2_final, b2_final = promediar_pesos(lista_pesos_entrenados)

    # ── PASO 5: Evaluar el modelo promediado ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  PASO 5: EVALUACIÓN DEL MODELO PROMEDIADO")
    print("=" * 60)

    y_pred_final = predecir(X_test, W1_final, b1_final, W2_final, b2_final)
    acc_final = precision(y_pred_final, y_test)
    print(f"\n  Precisión FINAL del modelo promediado en TEST: {acc_final:.2f}%")

    # ── Comparar con cada red individual ──────────────────────────────────────
    print(f"\n  Comparación con redes individuales:")
    for k, (W1_k, b1_k, W2_k, b2_k) in enumerate(lista_pesos_entrenados):
        y_pred_k = predecir(X_test, W1_k, b1_k, W2_k, b2_k)
        acc_k = precision(y_pred_k, y_test)
        print(f"    Red {k+1} individual: {acc_k:.2f}%")
    print(f"    Modelo promediado : {acc_final:.2f}%")

    # ── Calcular historial promedio para la gráfica ───────────────────────────
    # Promediamos las pérdidas y precisiones de todas las particiones por época
    # para tener UNA curva representativa del entrenamiento general
    #
    # historiales_loss es una lista de K listas, cada una de longitud `epocas`
    # np.array(historiales_loss) → forma (K, epocas)
    # np.mean(axis=0)           → forma (epocas,)  ← promedio por época
    historial_loss_promedio = np.mean(np.array(historiales_loss), axis=0).tolist()
    historial_acc_promedio  = np.mean(np.array(historiales_acc), axis=0).tolist()


    # ── Graficar resultados ────────────────────────────────────────────────
    graficar_arnovi(historiales_loss, historiales_acc,
                    historial_loss_promedio, historial_acc_promedio,
                    NUM_PARTICIONES)

    # Guardar el modelo entrenado
    tiempo_total = time.time() - tiempo_inicio
    
    y_pred_final = predecir(X_test, W1_final, b1_final, W2_final, b2_final)
    acc_final = precision(y_pred_final, y_test)
    print(f"\n  Tiempo total de entrenamiento: {tiempo_total:.2f}s")
    
    guardar_modelo(
        W1_final, b1_final, W2_final, b2_final,
        nombre_modelo='ArnoviNN',
        precision_test=acc_final,
        epocas=EPOCAS_POR_PARTICION,
        learning_rate=LEARNING_RATE,
        training_time=tiempo_total,
        info_extra={'num_particiones': NUM_PARTICIONES}
    )

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  ALGORITMO DE ARNOVI — RED NEURONAL DISTRIBUIDA — MNIST")
    print("  Sin librerías de ML, solo NumPy")
    print("=" * 60)

    # ── 1. Cargar el dataset ─────────────────────────────────────────────────
    X_all, y_all = cargar_mnist()

    # ── 2. Preprocesamos: aplanar, normalizar, one-hot, split 70/30 ──────────
    X_train, Y_train, y_train, X_test, Y_test, y_test = preprocesar(X_all, y_all)

    # ── 3. Ejecutar el Algoritmo de Arnovi ────────────────────────────────────
    #    NUM_PARTICIONES=5:  dividimos train en 5 subconjuntos
    #    epocas=100:         cada mini-red entrena 100 épocas
    #    lr=0.4:             learning rate igual que BasicNeuralNetwork
    
    entrenar_arnovi(
        X_train, Y_train, y_train,
        X_test, y_test,
        num_particiones=NUM_PARTICIONES,
        epocas=EPOCAS_POR_PARTICION,
        lr=LEARNING_RATE,
        intervalo_log=INTERVALO_LOG
    )

