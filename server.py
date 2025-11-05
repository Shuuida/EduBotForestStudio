"""
EduBot Backend Server
=================================

Servidor Flask para EduBot que integra:

 - Traducción bidireccional Python ⇄ Bloques
 - Modo estructural ML seguro (sin exec)
 - Ejecución de código Python aislada
 - Ejecución de flujos ML mixtos (MiniML / scikit-learn)
 - Exportación / Importación de archivos .edubotproj, .edubotml, .json
 - Gestión integral mediante file_handler

Versión: b1.2
"""

import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Dependencias internas del ecosistema EduBot
from core.ml_adapter import Translator, execute_structs
from core.ml_struct_rules import block_to_struct as ml_block_to_struct
from storage import ml_exporter
from storage import file_handler
from core.executor import execute_user_code

# ---------------------------------------------
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"status": "EduBot Backend Running", "version": "b1.2"}

# ============================================================
# TRADUCCIÓN BIDIRECCIONAL PYTHON ⇄ BLOQUES

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Expected JSON object"}), 400

    direction = data.get("direction")
    mode = data.get("mode", "code")  # code | struct

    if not direction:
        return jsonify({"error": "Missing 'direction'"}), 400

    translator = Translator()

    try:
        if mode == "struct" and direction == "to_struct":
            blocks = data.get("blocks", [])
            result = ml_exporter.export_blocks_to_struct(blocks)
            return jsonify(result), 200

        elif direction == "to_python":
            blocks = data.get("blocks", [])
            code = translator.translate_to_python(blocks)
            return jsonify({"result": code}), 200

        elif direction == "to_blocks":
            code = data.get("code", "")
            blocks = translator.translate_to_blocks(code)
            return jsonify({"result": blocks}), 200

        else:
            return jsonify({"error": "Invalid direction"}), 400

    except Exception as e:
        return jsonify({"error": f"Translation error: {str(e)}"}), 500

# ============================================================
# EJECUCIÓN DE CÓDIGO PYTHON

@app.route("/run", methods=["POST"])
def run():
    try:
        data = request.get_json(force=True)
        code = data.get("code", "")

        if not isinstance(code, str) or not code.strip():
            return jsonify({"error": "Invalid or empty code"}), 400

        result = execute_user_code(code)
        return jsonify({
            "success": result.get("success", False),
            "output": result.get("output", ""),
            "error": result.get("error", "")
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================
# EJECUCIÓN DE PIPELINES ML (bloques o estructuras)

@app.route("/ml_execute", methods=["POST"])
def ml_execute():
    try:
        data = request.get_json(force=True)
        blocks = data.get("blocks") or data.get("structs")

        if not blocks or not isinstance(blocks, list):
            return jsonify({"error": "No blocks/structs provided"}), 400

        structs = []
        for b in blocks:
            if isinstance(b, dict) and "action" in b:
                structs.append(b)
            else:
                structs.append(ml_block_to_struct(b))

        result = execute_structs(structs)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# EXPORTACIÓN ESTRUCTURAL (JSON / YAML)

@app.route("/export_struct", methods=["POST"])
def export_struct():
    try:
        data = request.get_json(force=True)
        blocks = data.get("blocks", [])
        export_format = data.get("format", "json")
        file_name = data.get("file_name", "ml_pipeline")

        struct_data = ml_exporter.export_blocks_to_struct(blocks)
        os.makedirs("exports", exist_ok=True)

        if export_format == "yaml":
            import yaml
            content = yaml.dump(struct_data, allow_unicode=True)
            path = f"exports/{file_name}.yaml"
        else:
            content = json.dumps(struct_data, indent=4, ensure_ascii=False)
            path = f"exports/{file_name}.json"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return jsonify({
            "success": True,
            "file": path,
            "data_preview": struct_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# FILE HANDLER — GESTIÓN UNIFICADA DE ARCHIVOS

@app.route("/file/save", methods=["POST"])
def save_file():
    try:
        data = request.get_json(force=True)
        name = data.get("name")
        content = data.get("content")
        success = file_handler.auto_export(content, name)
        return jsonify({"status": "saved" if success else "failed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/file/load", methods=["POST"])
def load_file():
    try:
        data = request.get_json(force=True)
        file_path = data.get("path")
        content = file_handler.auto_import(file_path)
        return jsonify({"content": content}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/file/list", methods=["GET"])
def list_files():
    try:
        files = {
            "projects": file_handler.list_projs(),
            "models": os.listdir("./models"),
            "datasets": os.listdir("./datasets")
        }
        return jsonify(files)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# STATUS (Diagnóstico rápido)

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "online",
        "modules": {
            "translator": True,
            "ml_adapter": True,
            "ml_exporter": True,
            "file_handler": True
        }
    })

# ============================================================
# MAIN ENTRY

if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    file_handler.ensure_dir_exist()
    app.run(host="127.0.0.1", port=5000, debug=True)
