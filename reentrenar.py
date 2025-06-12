# reentrenar.py

import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

DB_PATH = "tampu.db"
MODEL_PATH = "modelo_tampu.pkl"

# 1. Conectarse a la base de datos
conn = sqlite3.connect(DB_PATH)
query = "SELECT ECG, HRV, MOVIMIENTO, SpO2, prediccion FROM mediciones"
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Verificar si hay suficientes datos por clase
if df['prediccion'].nunique() < 3 or df.shape[0] < 30:
    print("⚠️ No hay suficientes datos diversos para reentrenar. Abortando.")
    exit()

# 3. Separar variables (X) y etiquetas (y)
X = df[['ECG', 'HRV', 'MOVIMIENTO', 'SpO2']]
y = df['prediccion']

# 4. Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 5. Entrenar el nuevo modelo
modelo = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluar y mostrar resultados
print("✅ Modelo reentrenado:")
print(classification_report(y_test, modelo.predict(X_test)))

# 7. Guardar el nuevo modelo
with open(MODEL_PATH, "wb") as f:
    pickle.dump(modelo, f)

print(f"🎉 Nuevo modelo guardado como '{MODEL_PATH}'")
