"""
=============================================================================
  RED NEURONAL DESDE CERO CON NUMPY — DATASET MNIST
=============================================================================

ARQUITECTURA
─────────────
  Capa de Entrada  : 784 neuronas  (28×28 píxeles aplanados a un vector)
        ↓
  Capa Oculta      : 128 neuronas  (ReLU como función de activación)
        ↓
  Capa de Salida   : 10 neuronas   (Softmax — una por dígito del 0 al 9)

  Cada época movemos los pesos un pequeño paso en la dirección que reduce
  el error. El learning_rate (0.4) controla el tamaño del paso.

=============================================================================
"""

import sys
import os
import numpy as np

# ── Agregar el directorio de módulos al path para encontrar los módulos ─────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Modules'))

from DatasetHandling import *
from Graphics import *
from Fuctions import *
from WeightsHandling import *
from ModelPersistence import guardar_modelo, cargar_modelo


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar(X_train, Y_train, y_train, X_test, y_test,
             epocas, lr, intervalo_log):
    """
    Una ÉPOCA = un ciclo completo sobre TODO el dataset de entrenamiento.
    Como usamos Gradient Descent full-batch (no estocástico), en cada época:
      1. Hacemos el forward pass con TODAS las muestras
      2. Calculamos la pérdida
      3. Hacemos el backward pass para obtener gradientes
      4. Actualizamos los pesos

    Esto contrasta con el Stochastic GD (SGD) donde usaríamos una sola
    muestra por actualización, o el Mini-Batch GD donde usaríamos grupos.

    Ventaja del full-batch: gradientes estables y exactos.
    Desventaja: más lento por época al procesar todas las muestras a la vez.
    """
    print("\n" + "="*60)
    print("  INICIALIZANDO PESOS")
    print("="*60)
    W1, b1, W2, b2 = inicializar_pesos()

    historial_loss = []
    historial_acc  = []

    print("\n" + "="*60)
    print("  INICIANDO ENTRENAMIENTO")
    print(f"  Épocas: {epocas}  |  Learning Rate: {lr}")
    print(f"  Muestras en entrenamiento: {X_train.shape[0]}")
    print("="*60)

    for epoca in range(1, epocas + 1):

        # ── Forward Pass ─────────────────────────────────────────────────────
        # Calculamos Z1, A1, Z2, A2 para TODAS las muestras de entrenamiento
        Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

        # ── Pérdida ──────────────────────────────────────────────────────────
        loss = cross_entropy(A2, Y_train)
        historial_loss.append(loss)

        # ── Precisión en entrenamiento ────────────────────────────────────────
        y_pred_train = np.argmax(A2, axis=1)
        acc_train = precision(y_pred_train, y_train)
        historial_acc.append(acc_train)

        # ── Backward Pass ────────────────────────────────────────────────────
        dW1, db1, dW2, db2 = backward(X_train, Y_train, Z1, A1, A2, W2)

        # ── Actualizar Pesos ─────────────────────────────────────────────────
        W1, b1, W2, b2 = actualizar_pesos(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        # ── Log de progreso ───────────────────────────────────────────────────
        if epoca % intervalo_log == 0 or epoca == 1:
            y_pred_test = predecir(X_test, W1, b1, W2, b2)
            acc_test = precision(y_pred_test, y_test)
            print(f"  Época {epoca:4d}/{epocas} │ "
                  f"Loss: {loss:.4f} │ "
                  f"Acc Train: {acc_train:.1f}% │ "
                  f"Acc Test: {acc_test:.1f}%")

    # ── Evaluación final ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUACIÓN FINAL")
    print("="*60)
    y_pred_test = predecir(X_test, W1, b1, W2, b2)
    acc_final = precision(y_pred_test, y_test)
    print(f"\n  Precisión final en TEST: {acc_final:.2f}%")

    # Graficamos el progreso del entrenamiento
    graficar_resultados(historial_loss, historial_acc)

    # Guardar el modelo entrenado
    y_pred_final = predecir(X_test, W1, b1, W2, b2)
    acc_final = precision(y_pred_final, y_test)
    guardar_modelo(
        W1, b1, W2, b2,
        nombre_modelo='BasicNN',
        precision_test=acc_final,
        epocas=100,
        learning_rate=0.4
    )

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  RED NEURONAL DESDE CERO — MNIST")
    print("  Sin librerías de ML, solo NumPy")
    print("="*60)

    # 1. Cargamos el dataset
    X_all, y_all = cargar_mnist()

    # 2. Preprocesamos: aplanar, normalizar, one-hot, split 70/30
    X_train, Y_train, y_train, X_test, Y_test, y_test = preprocesar(X_all, y_all)

    # 3. Entrenamos la red neuronal
    #    epocas=100: pasamos 100 veces por todos los datos
    #    lr=0.4:     learning rate que controla el tamaño del paso
    entrenar(
        X_train, Y_train, y_train,
        X_test, y_test,
        epocas=100,
        lr=0.4,
        intervalo_log=10
    )
