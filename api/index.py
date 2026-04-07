from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Cargar el modelo
# Asegúrate de que la ruta sea correcta relativa al archivo
model_path = os.path.join(os.path.dirname(__file__), 'HorseColic_Model.joblib')
modelo = joblib.load(model_path)

# Definir la estructura de los datos de entrada
class DatosCaballo(BaseModel):
    rectal_temperature: float
    pulse: float
    respiratory_rate: float
    # ... (Añade el resto de tus 15 variables finales aquí)

@app.post("/predecir")
def predecir_lesion(datos: DatosCaballo):
    # Convertir a DataFrame
    df = pd.DataFrame([datos.dict()])
    
    # Predecir
    prediccion = modelo.predict(df)[0]
    probabilidades = modelo.predict_proba(df)[0]
    
    resultado = "Sí requiere cirugía (1)" if prediccion == 1 else "Tratamiento Médico (2)"
    
    return {
        "prediccion": int(prediccion),
        "resultado_clinico": resultado,
        "probabilidad_cirugia": float(probabilidades[0]),
        "probabilidad_medico": float(probabilidades[1])
    }