"""
=============================================================================
  RED NEURONAL DESDE CERO CON NUMPY — DATASET MNIST
=============================================================================

GUÍA CONCEPTUAL ANTES DE LEER EL CÓDIGO
─────────────────────────────────────────
  Una red neuronal es básicamente una cadena de multiplicaciones de matrices
  seguidas de funciones no lineales (activaciones). El objetivo del
  entrenamiento es encontrar los valores de esas matrices (pesos W) y
  vectores (sesgos b) que hacen que la red clasifique correctamente.

ARQUITECTURA QUE USAREMOS
─────────────────────────
  Capa de Entrada  : 784 neuronas  (28×28 píxeles aplanados a un vector)
        ↓
  Capa Oculta      : 128 neuronas  (ReLU como función de activación)
        ↓
  Capa de Salida   : 10 neuronas   (Softmax — una por dígito del 0 al 9)

NOTACIÓN MATEMÁTICA
───────────────────
  X      → Matriz de entradas,  forma (N, 784)  donde N = número de muestras
  W1     → Pesos capa 1,        forma (784, 128)
  b1     → Sesgos capa 1,       forma (1, 128)
  W2     → Pesos capa 2,        forma (128, 10)
  b2     → Sesgos capa 2,       forma (1, 10)
  Y      → Etiquetas en one-hot, forma (N, 10)

FORWARD PASS (paso hacia adelante)
───────────────────────────────────
  Z1 = X  · W1 + b1          ← Combinación lineal (multiplicación de matrices)
  A1 = ReLU(Z1)              ← Activación: max(0, x)
  Z2 = A1 · W2 + b2          ← Segunda combinación lineal
  A2 = Softmax(Z2)           ← Activación: convierte en probabilidades (suman 1)

  La salida A2 es un vector de 10 probabilidades. Ejemplo: A2[3] = 0.92
  significa que la red está 92% segura de que el dígito es un 3.

FUNCIÓN DE PÉRDIDA (Loss)
─────────────────────────
  Usamos Entropía Cruzada (Cross-Entropy):
      L = -1/N · Σ Σ Y_ij · log(A2_ij)

  Mide qué tan lejos está la predicción de la realidad. Si la red predice
  correctamente con alta confianza, L es cercano a 0.

BACKWARD PASS (backpropagation — la magia del aprendizaje)
───────────────────────────────────────────────────────────
  Calculamos cuánto contribuyó cada peso al error usando la regla de la cadena
  del cálculo diferencial (derivadas parciales).

  dZ2 = A2 - Y                       ← Gradiente combinado softmax + cross-entropy
  dW2 = (1/N) · A1ᵀ · dZ2           ← Gradiente de los pesos W2
  db2 = (1/N) · Σ dZ2               ← Gradiente de los sesgos b2
  dA1 = dZ2 · W2ᵀ                   ← Error propagado hacia atrás
  dZ1 = dA1 * ReLU'(Z1)             ← ReLU'(x) = 1 si x>0, sino 0
  dW1 = (1/N) · Xᵀ · dZ1           ← Gradiente de los pesos W1
  db1 = (1/N) · Σ dZ1               ← Gradiente de los sesgos b1

ACTUALIZACIÓN DE PESOS (Gradient Descent)
──────────────────────────────────────────
  W = W - learning_rate · dW
  b = b - learning_rate · db

  Cada época movemos los pesos un pequeño paso en la dirección que reduce
  el error. El learning_rate (0.4) controla el tamaño del paso.

=============================================================================
"""

import sys
import os
import numpy as np

# ── Agregar el directorio padre al path para encontrar los módulos ─────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from DatasetHandling import *
from Graphics import *
from Fuctions import *
from WeightsHandling import *
from ModelPersistence import guardar_modelo, cargar_modelo, probar_modelo, probar_imagen


# ─────────────────────────────────────────────────────────────────────────────
# BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar(X_train, Y_train, y_train, X_test, y_test,
             epocas=100, lr=0.3, intervalo_log=10):
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

    return W1, b1, W2, b2, historial_loss, historial_acc


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
    W1, b1, W2, b2, historial_loss, historial_acc = entrenar(
        X_train, Y_train, y_train,
        X_test, y_test,
        epocas=100,
        lr=0.4,
        intervalo_log=10
    )

    # 4. Graficamos el progreso del entrenamiento
    graficar_resultados(historial_loss, historial_acc)

    # 5. Guardar el modelo entrenado
    y_pred_final = predecir(X_test, W1, b1, W2, b2)
    acc_final = precision(y_pred_final, y_test)
    guardar_modelo(
        W1, b1, W2, b2,
        nombre_modelo='BasicNN',
        precision_test=acc_final,
        epocas=100,
        learning_rate=0.4
    )
