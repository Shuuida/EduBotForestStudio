import eel
import json
import os

# Memoria temporal de la sesión actual
CURRENT_SESSION = {
    "student_id": None,
    "is_teacher": False
}

@eel.expose
def login_student(student_id):
    """Activa el perfil del estudiante en la sesión local."""
    if not student_id or len(student_id.strip()) == 0:
        return {"status": "error", "message": "El ID no puede estar vacío."}
    
    CURRENT_SESSION["student_id"] = student_id.strip()
    CURRENT_SESSION["is_teacher"] = False
    return {"status": "success", "message": f"Sesión iniciada como: {student_id}"}

@eel.expose
def get_current_student():
    """El Frontend llama a esto para saber qué nombre poner en la esquina superior."""
    return CURRENT_SESSION["student_id"]

@eel.expose
def logout_student():
    CURRENT_SESSION["student_id"] = None
    return {"status": "success"}