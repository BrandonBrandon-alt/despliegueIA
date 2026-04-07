from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="API Predictor de Cólico Equino")

# --- DIAGNÓSTICO TEMPORAL ---
# ruta_modelo = os.path.join(os.path.dirname(__file__), 'HorseColic_Model.joblib')
# modelo = joblib.load(ruta_modelo)

# Simulador de modelo (para ver si arranca el servidor)
class DummyModel:
    def predict(self, df): return [1]
    def predict_proba(self, df): return [[0.9, 0.1]]
    classes_ = [1, 2]

modelo = DummyModel()
# ----------------------------

# 2. Definir las 13 variables exactas que espera tu modelo
class DatosCaballo(BaseModel):
    age: float
    rectal_temperature: float
    pulse: float
    respiratory_rate: float
    temperature_extremities: float
    peripheral_pulse: float
    mucous_membranes: float
    capillary_refill_time: float
    pain: float
    peristalsis: float
    abdominal_distension: float
    packed_cell_volume: float
    total_protein: float

@app.get("/")
def read_root():
    return {
        "mensaje": "¡API de Predicción de Cólico Equino funcionando en Vercel!",
        "documentacion": "/docs",
        "metodo": "Usa POST en /predecir para obtener resultados"
    }

@app.post("/predecir")
def predecir_lesion(datos: DatosCaballo):
    # Convertir los datos recibidos a un DataFrame con una sola fila
    df = pd.DataFrame([datos.dict()])
    
    # Predecir usando el pipeline (que hace la limpieza y predicción automáticamente)
    prediccion = modelo.predict(df)[0]
    probabilidades = modelo.predict_proba(df)[0]
    
    # Tu modelo clasifica: 1 (Cirugía), 2 (Médico)
    resultado = "Sí requiere cirugía (1)" if prediccion == 1 else "Tratamiento Médico (2)"
    
    # El índice de probabilidad puede variar según el orden de las clases (1 o 2)
    # Por defecto scikit-learn ordena las clases de menor a mayor (1 primero, 2 segundo)
    prob_cirugia = probabilidades[0] if modelo.classes_[0] == 1 else probabilidades[1]
    
    return {
        "prediccion": int(prediccion),
        "resultado_clinico": resultado,
        "probabilidad_cirugia": round(float(prob_cirugia) * 100, 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)