from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="API Predictor de Cólico Equino")

# --- CARGA DEL MODELO REAL ---
try:
    # Ruta absoluta mejorada para Vercel
    base_path = os.path.dirname(__file__)
    ruta_modelo = os.path.join(base_path, 'HorseColic_Model.joblib')
    
    # Cargar el modelo
    if os.path.exists(ruta_modelo):
        modelo = joblib.load(ruta_modelo)
        status_modelo = "Modelo real cargado con éxito"
    else:
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_modelo}")
        
except Exception as e:
    # Si falla, mantenemos el dummy PERO avisamos claramente el error
    class DummyModel:
        def predict(self, df): return [1]
        def predict_proba(self, df): return [[0.9, 0.1]]
        classes_ = [1, 2]
    modelo = DummyModel()
    status_modelo = f"ERROR DE CARGA: {str(e)}"
# ----------------------------

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
        "mensaje": "¡API de Predicción de Cólico Equino!",
        "status_modelo": status_modelo,
        "instrucciones": "Ve a /docs para probar el modelo"
    }

@app.post("/predecir")
def predecir_lesion(datos: DatosCaballo):
    df = pd.DataFrame([datos.dict()])
    
    try:
        prediccion = modelo.predict(df)[0]
        probabilidades = modelo.predict_proba(df)[0]
        
        resultado = "Sí requiere cirugía (1)" if prediccion == 1 else "Tratamiento Médico (2)"
        
        # Ajuste de probabilidad
        idx = 0 if modelo.classes_[0] == 1 else 1
        prob_cirugia = probabilidades[idx]
        
        return {
            "status_modelo": status_modelo,
            "prediccion": int(prediccion),
            "resultado_clinico": resultado,
            "probabilidad_cirugia": round(float(prob_cirugia) * 100, 2)
        }
    except Exception as e:
        return {"error": f"Fallo en predicción: {str(e)}", "status_modelo": status_modelo}