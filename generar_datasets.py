import os
import csv
import random

# Escenario: Clasificador de Reciclaje para Arduino Uno
# Features: [Sensor Inductivo (0-1023), Peso (gramos), Brillo/Reflexión (0-1023)]
# Target: 0=Cartón, 1=Plástico, 2=Metal

reciclaje_data = [["inductivo", "peso_g", "brillo", "material"]]

# Generamos datos sintéticos con "ruido" realista para que el RF tenga sentido

# 1. CARTÓN (Clase 0): No detecta metal, ligero, mate.
for _ in range(15):
    ind = random.randint(0, 50)       # El sensor inductivo casi no reacciona
    peso = random.randint(10, 40)     # Ligero
    brillo = random.randint(100, 300) # Opaco/Mate
    reciclaje_data.append([ind, peso, brillo, 0])

# 2. PLÁSTICO (Clase 1): No detecta metal, peso variado, brillante.
for _ in range(15):
    ind = random.randint(0, 60)        # No reacciona al metal
    peso = random.randint(30, 80)      # Más denso que el cartón
    brillo = random.randint(400, 750)  # Refleja luz (superficie lisa)
    reciclaje_data.append([ind, peso, brillo, 1])

# 3. METAL (Clase 2): Detecta metal, variado peso, muy brillante.
for _ in range(15):
    ind = random.randint(800, 1023)    # ¡Disparo del sensor inductivo!
    peso = random.randint(20, 150)     # Desde latas ligeras a tornillos pesados
    brillo = random.randint(700, 1000) # Muy brillante
    reciclaje_data.append([ind, peso, brillo, 2])

# Casos "Difíciles" (Donde ML brilla vs IFs simples)
# - Plástico pesado que parece metal por peso pero no por inducción
reciclaje_data.append([40, 120, 600, 1]) 
# - Lata de aluminio muy ligera (parece cartón por peso, pero metal por inducción)
reciclaje_data.append([950, 15, 850, 2])
# - Cartón con cinta adhesiva brillante (parece plástico por brillo)
reciclaje_data.append([20, 35, 650, 0])


def create_dataset(filename, data):
    folder = "datasets"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    try:
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"✅ Generado: {path} (Escenario Arduino Reciclaje)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_dataset("reciclaje.csv", reciclaje_data)