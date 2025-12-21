import os
import csv
import random

# Escenario: Termostato Inteligente con Red Neuronal
# La NN aprende a predecir el "Índice de Calor" sin usar la fórmula pesada.
# Inputs normalizados (0-1) para facilitar el entrenamiento en el Arduino.

def calcular_heat_index(T, R):
    # Fórmula simplificada de Rothfusz (en Fahrenheit para el cálculo interno)
    Tf = T * 9/5 + 32
    HI = -42.379 + 2.04901523*Tf + 10.14333127*R - 0.22475541*Tf*R \
         - 0.00683783*Tf*Tf - 0.05481717*R*R + 0.00122874*Tf*Tf*R \
         + 0.00085282*Tf*R*R - 0.00000199*Tf*Tf*R*R
    
    # Convertir a Celsius
    return (HI - 32) * 5/9

confort_data = [["temp_norm", "hum_norm", "ac_on"]]

# Generamos 100 muestras de situaciones variadas
for _ in range(100):
    # Generar valores reales
    temp_real = random.uniform(20, 40)  # Entre 20°C y 40°C
    hum_real = random.uniform(30, 90)   # Entre 30% y 90%
    
    # Calcular sensación térmica real
    sensacion = calcular_heat_index(temp_real, hum_real)
    
    # Decisión: ¿Encender Aire? (Umbral de confort: 27°C sensación)
    # Esta frontera es curva (no lineal), perfecta para una NN.
    target = 1 if sensacion > 27.0 else 0
    
    # Normalizar inputs para la NN (CRÍTICO para MiniML)
    # Mapeamos los rangos a 0.0 - 1.0
    t_norm = (temp_real - 20) / (40 - 20)
    h_norm = (hum_real - 30) / (90 - 30)
    
    confort_data.append([round(t_norm, 4), round(h_norm, 4), target])

# Agregar casos de borde manuales para asegurar aprendizaje
# - Calor seco (35°C, 30% Hum) -> Confortable/Límite -> 0
confort_data.append([0.75, 0.0, 0]) 
# - Calor húmedo (30°C, 85% Hum) -> Sofocante -> 1
confort_data.append([0.5, 0.91, 1])

def create_dataset(filename, data):
    folder = "datasets"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    try:
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"✅ Generado: {path} (Escenario Smart HVAC)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_dataset("confort.csv", confort_data)