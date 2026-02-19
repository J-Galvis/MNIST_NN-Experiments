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

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: DESCARGA Y PREPARACIÓN DEL DATASET MNIST
# ─────────────────────────────────────────────────────────────────────────────
# Usamos torchvision SOLO para descargar el dataset. Inmediatamente
# convertimos todo a arrays de NumPy y no volvemos a usar PyTorch.

def cargar_mnist():
    """
    Descarga MNIST con torchvision y lo convierte completamente a NumPy.
    Retorna X (imágenes) e y (etiquetas) como arrays de NumPy.
    """
    print("Descargando MNIST con torchvision...")
    try:
        from torchvision import datasets
        from torchvision import transforms

        # Descargamos train y test por separado (estándar MNIST)
        transform = transforms.ToTensor()  # solo para poder acceder a los datos
        mnist_train = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
        mnist_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Convertimos a numpy de inmediato
        # .data son los píxeles: tensor de forma (60000, 28, 28) con valores 0-255
        # .targets son las etiquetas: tensor de forma (60000,) con valores 0-9
        X_train_raw = mnist_train.data.numpy().astype(np.float64)   # (60000, 28, 28)
        y_train_raw = mnist_train.targets.numpy()                    # (60000,)
        X_test_raw  = mnist_test.data.numpy().astype(np.float64)    # (10000, 28, 28)
        y_test_raw  = mnist_test.targets.numpy()                     # (10000,)

        # Unimos todo en un solo bloque (70000 muestras totales)
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)   # (70000, 28, 28)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)   # (70000,)

        print(f"  Dataset completo: {X_all.shape[0]} imágenes de 28×28 píxeles")
        return X_all, y_all

    except ImportError:
        raise ImportError(
            "No se encontró torchvision. Instálalo con:\n"
            "  pip install torch torchvision"
        )


def preprocesar(X_all, y_all, fraccion_entrenamiento=0.7):
    """
    Aplana, normaliza y divide el dataset.

    Aplanar: (N, 28, 28) → (N, 784)
      Cada imagen 2D se convierte en un vector 1D de 784 elementos.
      Esto nos da la matriz X que necesitamos para multiplicar con W1.

    Normalizar: dividir por 255
      Los píxeles van de 0 a 255. Dividimos por 255 para que queden entre
      0 y 1. Esto ayuda al gradiente descendente a converger más rápido
      porque los valores no son demasiado grandes.

    One-Hot Encoding:
      Si la etiqueta es el dígito 3, la convertimos al vector:
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
      Esto es necesario para calcular la pérdida con Softmax.
    """
    N = X_all.shape[0]

    # ── Aplanar ──────────────────────────────────────────────────────────────
    # reshape(-1, 784): -1 significa "NumPy calcula este eje automáticamente"
    # Básicamente decimos: "convierte cada imagen 28×28 en una fila de 784"
    X = X_all.reshape(-1, 784)          # (70000, 784)

    # ── Normalizar ───────────────────────────────────────────────────────────
    X = X / 255.0                       # valores entre 0 y 1

    # ── One-Hot Encoding ─────────────────────────────────────────────────────
    # np.eye(10) crea la matriz identidad 10×10
    # Si y[i] = 3, entonces np.eye(10)[3] = [0,0,0,1,0,0,0,0,0,0]
    Y = np.eye(10)[y_all]               # (70000, 10)

    # ── División 70% entrenamiento / 30% prueba ───────────────────────────────
    n_train = int(N * fraccion_entrenamiento)
    # Mezclamos aleatoriamente los índices para que el split sea representativo
    indices = np.random.permutation(N)
    idx_train = indices[:n_train]
    idx_test  = indices[n_train:]

    X_train, Y_train, y_train = X[idx_train], Y[idx_train], y_all[idx_train]
    X_test,  Y_test,  y_test  = X[idx_test],  Y[idx_test],  y_all[idx_test]

    print(f"\n  Muestras de entrenamiento : {X_train.shape[0]}")
    print(f"  Muestras de prueba        : {X_test.shape[0]}")
    print(f"  Forma de X_train          : {X_train.shape}  ← (muestras, píxeles)")
    print(f"  Forma de Y_train          : {Y_train.shape}  ← (muestras, clases)")

    return X_train, Y_train, y_train, X_test, Y_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: FUNCIONES DE ACTIVACIÓN Y PÉRDIDA
# ─────────────────────────────────────────────────────────────────────────────

def relu(Z):
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x)

    ¿Por qué necesitamos activaciones?
    Sin ellas, la red sería equivalente a UNA sola multiplicación de matrices
    sin importar cuántas capas tenga (composición de funciones lineales = lineal).
    ReLU introduce no-linealidad, permitiendo a la red aprender patrones complejos.

    Ejemplo con una fila: [-2.5, 0.0, 3.1, -0.4, 1.7]
    Después de ReLU:       [ 0.0, 0.0, 3.1,  0.0, 1.7]
    """
    return np.maximum(0, Z)


def derivada_relu(Z):
    """
    Derivada de ReLU:
      ReLU'(x) = 1  si x > 0
      ReLU'(x) = 0  si x ≤ 0

    En NumPy: (Z > 0) retorna True/False, que se comporta como 1/0
    en operaciones aritméticas.

    Usamos esto en el backward pass para saber qué neuronas "se activaron"
    y cuáles no contribuyeron al error.
    """
    return (Z > 0).astype(np.float64)


def softmax(Z):
    """
    Softmax: convierte un vector de números reales en probabilidades que suman 1.

      Softmax(z_i) = exp(z_i) / Σ exp(z_j)

    Por qué restamos el máximo (truco de estabilidad numérica):
      exp(700) desborda a infinito en float64. Si restamos el máximo primero,
      los valores quedan en rango manejable sin cambiar el resultado matemático
      porque exp(a-c) / Σ exp(b-c) = exp(a) / Σ exp(b).

    Z tiene forma (N, 10). Procesamos todas las muestras a la vez.
    keepdims=True mantiene las dimensiones para que la resta funcione bien.
    """
    Z_estable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_estable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def cross_entropy(A2, Y):
    """
    Entropía Cruzada: L = -1/N · Σ Σ Y_ij · log(A2_ij)

    Mide la diferencia entre la distribución predicha (A2) y la real (Y).
    Cuando la predicción es perfecta (A2[clase_correcta] ≈ 1), la pérdida → 0.
    Cuando la predicción es terrible (A2[clase_correcta] ≈ 0), la pérdida → ∞.

    np.clip evita log(0) = -∞ que haría explotar los cálculos.
    """
    N = Y.shape[0]
    A2_seguro = np.clip(A2, 1e-15, 1 - 1e-15)
    return -np.sum(Y * np.log(A2_seguro)) / N


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3: INICIALIZACIÓN DE PESOS
# ─────────────────────────────────────────────────────────────────────────────

def inicializar_pesos(semilla=42):
    """
    ¿Por qué no inicializar todos los pesos en 0?
    Si todos los pesos son iguales, todas las neuronas de una capa aprenden
    exactamente lo mismo (problema de simetría). Necesitamos valores aleatorios
    para que cada neurona aprenda características distintas.

    Inicialización de He (para ReLU):
      W = N(0, 1) * sqrt(2 / n_entradas)
    Escala los pesos para que la varianza de las activaciones sea estable
    durante el forward pass. Sin esto, los gradientes pueden explotar o
    desvanecerse con redes profundas.
    """
    np.random.seed(semilla)

    # W1: forma (784, 128) — conecta entrada con capa oculta
    # Cada columna j contiene los pesos que van HACIA la neurona j de la capa oculta
    W1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)

    # b1: forma (1, 128) — un sesgo por neurona en la capa oculta
    # Los sesgos los iniciamos en 0 (no tienen el problema de simetría)
    b1 = np.zeros((1, 128))

    # W2: forma (128, 10) — conecta capa oculta con capa de salida
    W2 = np.random.randn(128, 10) * np.sqrt(2.0 / 128)

    # b2: forma (1, 10) — un sesgo por clase de salida
    b2 = np.zeros((1, 10))

    print(f"\n  W1: {W1.shape}  (pesos entrada → oculta)")
    print(f"  b1: {b1.shape}  (sesgos capa oculta)")
    print(f"  W2: {W2.shape}  (pesos oculta → salida)")
    print(f"  b2: {b2.shape}  (sesgos capa salida)")

    return W1, b1, W2, b2


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4: FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────

def forward(X, W1, b1, W2, b2):
    """
    Propaga la entrada X a través de la red para obtener predicciones.

    Álgebra lineal involucrada:
    ───────────────────────────
    Z1 = X · W1 + b1
      X  es (N, 784)
      W1 es (784, 128)
      X·W1 es (N, 128)   ← regla de multiplicación de matrices: (m,n)·(n,p)=(m,p)
      b1 es (1, 128)     ← se suma con broadcasting: NumPy expande b1 a (N,128)

    Cada fila de Z1 corresponde a una muestra. El elemento Z1[i,j] es la
    suma ponderada de todos los píxeles de la muestra i hacia la neurona j.
    En otras palabras: Z1[i,j] = Σ_k X[i,k] * W1[k,j] + b1[0,j]

    Retornamos Z1 y Z2 porque los necesitamos en el backward pass
    para calcular las derivadas de ReLU.
    """
    # ── Capa 1 ───────────────────────────────────────────────────────────────
    Z1 = X @ W1 + b1        # (N, 784) @ (784, 128) + (1, 128) = (N, 128)
    A1 = relu(Z1)           # (N, 128)  — aplicamos ReLU elemento a elemento

    # ── Capa 2 ───────────────────────────────────────────────────────────────
    Z2 = A1 @ W2 + b2       # (N, 128) @ (128, 10)  + (1, 10)  = (N, 10)
    A2 = softmax(Z2)        # (N, 10)  — convertimos a probabilidades

    return Z1, A1, Z2, A2


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5: BACKWARD PASS (Backpropagation)
# ─────────────────────────────────────────────────────────────────────────────

def backward(X, Y, Z1, A1, A2, W2):
    """
    Calcula los gradientes usando la regla de la cadena (chain rule).

    Idea intuitiva:
    ───────────────
    Queremos saber: "si aumento un poco W2[i,j], ¿sube o baja la pérdida?"
    Eso es la derivada parcial ∂L/∂W2[i,j], o sea el gradiente dW2.

    Con la regla de la cadena:
      ∂L/∂W2 = ∂L/∂A2 · ∂A2/∂Z2 · ∂Z2/∂W2

    El truco bonito de Softmax + Cross-Entropy es que sus derivadas
    combinadas simplifican a:
      dZ2 = A2 - Y

    Esto significa: el error es simplemente "qué tan diferente fue la
    probabilidad predicha de la etiqueta real". Elegante y eficiente.

    Dimensiones (verificación de álgebra):
    ───────────────────────────────────────
    dZ2  : (N, 10)
    dW2  : (128, N) @ (N, 10)  =  (128, 10)  ← misma forma que W2 ✓
    db2  : promedio de (N, 10) sobre eje 0    =  (1, 10)   ← misma forma que b2 ✓
    dA1  : (N, 10) @ (10, 128) =  (N, 128)
    dZ1  : (N, 128) * (N, 128) =  (N, 128)   ← multiplicación elemento a elemento
    dW1  : (784, N) @ (N, 128) =  (784, 128) ← misma forma que W1 ✓
    db1  : promedio de (N, 128) sobre eje 0   =  (1, 128)  ← misma forma que b1 ✓
    """
    N = X.shape[0]

    # ── Gradientes de la capa de salida ───────────────────────────────────────
    dZ2 = A2 - Y                              # (N, 10)  — error de predicción
    dW2 = A1.T @ dZ2 / N                     # (128, 10) — promedio del gradiente
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N  # (1, 10)

    # ── Propagamos el error hacia atrás a través de W2 ────────────────────────
    dA1 = dZ2 @ W2.T                          # (N, 128) — error en capa oculta

    # ── Pasamos a través de la derivada de ReLU ───────────────────────────────
    # Solo las neuronas que estaban "activas" (Z1 > 0) reciben gradiente
    dZ1 = dA1 * derivada_relu(Z1)            # (N, 128)

    # ── Gradientes de la capa de entrada ──────────────────────────────────────
    dW1 = X.T @ dZ1 / N                      # (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N  # (1, 128)

    return dW1, db1, dW2, db2


# ─────────────────────────────────────────────────────────────────────────────
# PASO 6: ACTUALIZACIÓN DE PESOS (Gradient Descent)
# ─────────────────────────────────────────────────────────────────────────────

def actualizar_pesos(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    """
    Gradient Descent (Descenso del Gradiente):
      W_nuevo = W_viejo - lr · dW

    Intuición:
      El gradiente apunta hacia la dirección de mayor aumento de la pérdida.
      Restándolo, vamos en la dirección opuesta: hacia donde la pérdida baja.
      lr (learning rate) controla qué tan grande es cada paso.

    ¿Por qué lr=0.4?
      - Si lr es muy grande: los pasos son enormes y el modelo "rebota" sin converger.
      - Si lr es muy pequeño: el aprendizaje es muy lento.
      - 0.4 es un valor razonable para este problema con full-batch GD.
    """
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


# ─────────────────────────────────────────────────────────────────────────────
# PASO 7: FUNCIÓN DE PREDICCIÓN Y PRECISIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predecir(X, W1, b1, W2, b2):
    """
    Realiza el forward pass y toma la clase con mayor probabilidad.
    np.argmax devuelve el índice del valor máximo en cada fila.
    Si A2[i] = [0.01, 0.02, 0.05, 0.85, ...], argmax = 3 → predicción: dígito 3
    """
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)


def precision(y_pred, y_real):
    """
    Porcentaje de predicciones correctas.
    np.mean(y_pred == y_real) compara elemento a elemento y promedia los True/False.
    """
    return np.mean(y_pred == y_real) * 100


# ─────────────────────────────────────────────────────────────────────────────
# PASO 8: BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def entrenar(X_train, Y_train, y_train, X_test, y_test,
             epocas=100, lr=0.4, intervalo_log=10):
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
# PASO 9: VISUALIZACIÓN OPCIONAL (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def graficar_resultados(historial_loss, historial_acc):
    """
    Grafica la pérdida y la precisión durante el entrenamiento.
    Requiere matplotlib, que es solo para visualización, no para la red.
    """
    try:
        import matplotlib.pyplot as plt

        epocas = range(1, len(historial_loss) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epocas, historial_loss, 'b-', linewidth=2)
        ax1.set_title('Pérdida (Loss) durante el entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.grid(True, alpha=0.3)

        ax2.plot(epocas, historial_acc, 'g-', linewidth=2)
        ax2.set_title('Precisión (Train Accuracy) durante el entrenamiento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión (%)')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('entrenamiento.png', dpi=120)
        plt.show()
        print("\n  Gráfica guardada en 'entrenamiento.png'")

    except ImportError:
        print("\n  (matplotlib no disponible, se omite la gráfica)")


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

    print("\n  ¡Entrenamiento completado!")
    print("  Los parámetros finales son: W1, b1, W2, b2")