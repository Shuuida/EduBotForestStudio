import eel
import csv
import os
from .autograder import _load_db

TEACHER_PIN = "1234" # En el futuro, esto se configura en la app

@eel.expose
def verify_teacher(pin):
    """Desbloquea el acceso al Dashboard local."""
    if pin == TEACHER_PIN:
        return {"status": "success", "message": "Acceso concedido"}
    return {"status": "error", "message": "PIN incorrecto"}

@eel.expose
def get_dashboard_data():
    """Devuelve todas las notas para pintarlas en una tabla de React."""
    db = _load_db()
    return {"status": "success", "data": db["grades"]}

@eel.expose
def export_to_excel(export_path="C:/Documents/edubot_notas.csv"):
    """
    Convierte el JSON a un archivo CSV que Excel abre nativamente.
    Se usa CSV nativo para mantener el backend de EduBot ligero.
    """
    db = _load_db()
    grades = db["grades"]
    
    if not grades:
        return {"status": "error", "message": "No hay datos para exportar."}

    try:
        # Asegurarse de que el directorio exista
        directory = os.path.dirname(export_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Crear el archivo CSV
        with open(export_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';') # Punto y coma evita problemas en Excel español
            
            # Escribir Cabeceras
            writer.writerow(["ID Estudiante", "Reto", "Aprobado", "Respuesta del Estudiante", "Fecha y Hora"])
            
            # Escribir Datos
            for record in grades:
                aprobado_str = "SÍ" if record["passed"] else "NO"
                writer.writerow([
                    record["student_id"], 
                    record["challenge"], 
                    aprobado_str, 
                    record["student_output"], 
                    record["timestamp"]
                ])
                
        return {"status": "success", "message": f"Notas exportadas exitosamente a {export_path}"}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}