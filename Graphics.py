import numpy as np
import matplotlib.pyplot as plt

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
# GRÁFICA EXTENDIDA PARA EL ALGORITMO DE ARNOVI
# ─────────────────────────────────────────────────────────────────────────────

def graficar_arnovi(historiales_loss, historiales_acc,
                    historial_loss_prom, historial_acc_prom,
                    num_particiones):
    """
    Genera gráficas que muestran:
      - La pérdida y precisión de CADA partición (curvas individuales)
      - La pérdida y precisión PROMEDIO (curva gruesa)

    Esto permite visualizar cómo cada mini-red aprende de forma diferente
    y cómo el promedio captura el comportamiento general.
    """
    try:
        import matplotlib.pyplot as plt

        epocas = range(1, len(historial_loss_prom) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ── Gráfica de pérdida ────────────────────────────────────────────────
        for k in range(num_particiones):
            ax1.plot(epocas, historiales_loss[k], alpha=0.4, linewidth=1,
                     label=f'Partición {k+1}')
        ax1.plot(epocas, historial_loss_prom, 'k-', linewidth=2.5,
                 label='Promedio', zorder=10)
        ax1.set_title('Pérdida (Loss) — Algoritmo de Arnovi')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Gráfica de precisión ──────────────────────────────────────────────
        for k in range(num_particiones):
            ax2.plot(epocas, historiales_acc[k], alpha=0.4, linewidth=1,
                     label=f'Partición {k+1}')
        ax2.plot(epocas, historial_acc_prom, 'k-', linewidth=2.5,
                 label='Promedio', zorder=10)
        ax2.set_title('Precisión (Train Acc) — Algoritmo de Arnovi')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión (%)')
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('entrenamiento_arnovi.png', dpi=120)
        plt.show()
        print("\n  Gráfica guardada en 'entrenamiento_arnovi.png'")

    except ImportError:
        print("\n  (matplotlib no disponible, se omite la gráfica)")

