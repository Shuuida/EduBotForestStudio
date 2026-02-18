import eel
import json
import os
from datetime import datetime
from .auth import CURRENT_SESSION

DB_PATH = "data/local_grades.json"

def _load_db():
    if not os.path.exists(DB_PATH):
        os.makedirs("data", exist_ok=True)
        with open(DB_PATH, "w") as f:
            json.dump({"grades": []}, f)
    with open(DB_PATH, "r") as f:
        return json.load(f)

def _save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)

@eel.expose
def submit_challenge(challenge_name, student_result, expected_result):
    """Evalúa el reto y guarda la nota localmente."""
    student_id = CURRENT_SESSION["student_id"]
    
    if not student_id:
        return {"status": "error", "message": "Debes iniciar sesión con tu ID primero."}

    # Lógica de validación (Autograding)
    # Se convierte a string para evitar errores de tipo (ej. 1 vs "1")
    passed = str(student_result).strip().lower() == str(expected_result).strip().lower()
    
    db = _load_db()
    
    # Crear el registro
    record = {
        "student_id": student_id,
        "challenge": challenge_name,
        "passed": passed,
        "student_output": student_result,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    db["grades"].append(record)
    _save_db(db)
    
    if passed:
        return {"status": "success", "passed": True, "message": "¡Reto superado! Excelente trabajo."}
    else:
        return {"status": "success", "passed": False, "message": "Resultado incorrecto. Sigue intentando."}