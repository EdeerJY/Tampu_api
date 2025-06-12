# main.py - Backend completo para Tampu con FastAPI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from datetime import datetime
import sqlite3
import os

app = FastAPI(title="Tampu API", description="API para predicción de ansiedad con ML y almacenamiento de historial.", version="1.0")

# ---------------------------
# 1. Cargar modelo ML
# ---------------------------
MODEL_PATH = "modelo_tampu.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No se encuentra el modelo .pkl")

with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

# ---------------------------
# 2. Conectar a base de datos SQLite
# ---------------------------
DB_PATH = "tampu.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS mediciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    ECG REAL,
    HRV REAL,
    MOVIMIENTO REAL,
    SpO2 REAL,
    prediccion INTEGER,
    timestamp TEXT
)''')

conn.commit()

# Crear tabla reentrenamientos si no existe
cursor.execute('''
CREATE TABLE IF NOT EXISTS reentrenamientos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT
)
''')
conn.commit()

# ---------------------------
# 3. Esquemas de entrada/salida
# ---------------------------
class Medidas(BaseModel):
    user_id: str
    ECG: float
    HRV: float
    MOVIMIENTO: float
    SpO2: float

class Resultado(BaseModel):
    prediccion: int
    interpretacion: str

# ---------------------------
# 4. Endpoint /predict
# ---------------------------
@app.post("/predict", response_model=Resultado)
def predecir(data: Medidas):
    try:
        valores = np.array([[data.ECG, data.HRV, data.MOVIMIENTO, data.SpO2]])
        pred = modelo.predict(valores)[0]

        interpretacion = {
            0: "Tranquilo",
            1: "Ansiedad leve",
            2: "Ansiedad fuerte"
        }.get(pred, "Desconocido")

        timestamp = datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO mediciones (user_id, ECG, HRV, MOVIMIENTO, SpO2, prediccion, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (data.user_id, data.ECG, data.HRV, data.MOVIMIENTO, data.SpO2, pred, timestamp)
        )
        conn.commit()

        # Verificar si hay suficientes datos para reentrenar
        cursor.execute("SELECT MAX(timestamp) FROM reentrenamientos")
        ultima_fecha = cursor.fetchone()[0] or "2000-01-01T00:00:00"

        cursor.execute("""
            SELECT COUNT(*) FROM mediciones
            WHERE timestamp > ?
        """, (ultima_fecha,))
        nuevas = cursor.fetchone()[0]

        if nuevas >= 100:
            print("🔁 Ejecutando reentrenamiento automático...")
            os.system("python reentrenar.py")
            cursor.execute("INSERT INTO reentrenamientos (timestamp) VALUES (?)", (datetime.utcnow().isoformat(),))
            conn.commit()

        return Resultado(prediccion=pred, interpretacion=interpretacion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# 5. Endpoint /historial/{user_id}
# ---------------------------
@app.get("/historial/{user_id}")
def historial(user_id: str):
    cursor.execute("""
        SELECT ECG, HRV, MOVIMIENTO, SpO2, prediccion, timestamp FROM mediciones
        WHERE user_id = ? ORDER BY timestamp DESC
    """, (user_id,))
    rows = cursor.fetchall()
    return [
        {
            "ECG": row[0],
            "HRV": row[1],
            "MOVIMIENTO": row[2],
            "SpO2": row[3],
            "prediccion": row[4],
            "timestamp": row[5]
        } for row in rows
    ]

# ---------------------------
# 6. Ejecutar local o en Render
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
