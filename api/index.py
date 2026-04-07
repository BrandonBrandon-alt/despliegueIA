from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os

app = FastAPI(title="API Predictor de Cólico Equino")

# --- CARGA DEL MODELO REAL ---
try:
    base_path = os.path.dirname(__file__)
    ruta_modelo = os.path.join(base_path, 'HorseColic_Model.joblib')
    
    if os.path.exists(ruta_modelo):
        modelo = joblib.load(ruta_modelo)
        status_modelo = "Modelo real cargado con éxito"
    else:
        status_modelo = f"ERROR: No se encontró el archivo en {ruta_modelo}"
        
except Exception as e:
    class DummyModel:
        def predict(self, df): return [1]
        def predict_proba(self, df): return [[0.9, 0.1]]
        classes_ = [1, 2]
    modelo = DummyModel()
    status_modelo = f"ERROR DE CARGA: {str(e)}"

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

# --- INTERFAZ HTML COOL ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictor de Cólico Equino | AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #a855f7;
            --bg: #0f172a;
            --card: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --danger: #ef4444;
            --success: #22c55e;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Outfit', sans-serif;
            background: radial-gradient(circle at top left, #1e1b4b, #0f172a);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 1100px;
            background: var(--card);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        header { text-align: center; margin-bottom: 40px; }
        h1 { font-size: 2.5rem; font-weight: 800; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        p.subtitle { color: #94a3b8; font-size: 1.1rem; margin-top: 8px; }

        .main-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 40px; }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

        .form-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 600px) { .form-section { grid-template-columns: 1fr; } }

        .input-group { display: flex; flex-direction: column; gap: 8px; }
        label { font-size: 0.9rem; font-weight: 500; color: #cbd5e1; }
        input, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 12px 16px;
            border-radius: 12px;
            color: white;
            font-size: 1rem;
            transition: all 0.3s;
        }
        input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2); }

        .btn-predict {
            grid-column: 1 / -1;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 16px;
            border: none;
            border-radius: 14px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4); }
        .btn-predict:active { transform: translateY(0); }

        .result-section {
            background: rgba(15, 23, 42, 0.4);
            border-radius: 20px;
            padding: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .result-badge {
            font-size: 1.5rem;
            font-weight: 700;
            padding: 10px 24px;
            border-radius: 100px;
            margin-bottom: 20px;
            display: none;
        }
        .requires-surgery { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
        .medical-treatment { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }

        .prob-container { width: 100%; margin-top: 20px; }
        .prob-bar-bg { width: 100%; height: 12px; background: #334155; border-radius: 100px; overflow: hidden; margin-bottom: 10px; }
        .prob-bar-fill { width: 0%; height: 100%; background: linear-gradient(to right, var(--primary), var(--secondary)); transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
        .prob-text { font-size: 3rem; font-weight: 800; }
        .prob-label { color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; font-size: 0.8rem; }

        .status-footer { margin-top: 30px; text-align: center; font-size: 0.8rem; color: #64748b; }
        .success-dot { color: var(--success); margin-right: 5px; }
        .error-dot { color: var(--danger); margin-right: 5px; }

        #loading { display: none; margin-bottom: 20px; }
        .spinner { width: 30px; height: 30px; border: 3px solid rgba(255, 255, 255, 0.1); border-top-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>HorseColic AI</h1>
            <p class="subtitle">Predictor de Intervención Quirúrgica en Cólico Equino</p>
        </header>

        <div class="main-grid">
            <form id="predictForm" class="form-section">
                <!-- Información General -->
                <div class="input-group">
                    <label>Edad del Caballo</label>
                    <select name="age" required>
                        <option value="1">Adulto</option>
                        <option value="2">Joven (< 6 meses)</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Temperatura Rectal (°C)</label>
                    <input type="number" step="0.1" name="rectal_temperature" value="38.5" required>
                </div>
                <!-- Signos Vitales -->
                <div class="input-group">
                    <label>Pulso (LPM)</label>
                    <input type="number" name="pulse" value="60" required>
                </div>
                <div class="input-group">
                    <label>Frec. Respiratoria</label>
                    <input type="number" name="respiratory_rate" value="20" required>
                </div>
                <!-- Examen Físico -->
                <div class="input-group">
                    <label>Temperatura Extremidades</label>
                    <select name="temperature_extremities" required>
                        <option value="1">Normal</option>
                        <option value="2">Cálida</option>
                        <option value="3">Fría</option>
                        <option value="4">Muy Fría</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Pulso Periférico</label>
                    <select name="peripheral_pulse" required>
                        <option value="1">Normal</option>
                        <option value="2">Aumentado</option>
                        <option value="3">Reducido</option>
                        <option value="4">Ausente</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Membranas Mucosas</label>
                    <select name="mucous_membranes" required>
                        <option value="1">Rosa Normal</option>
                        <option value="2">Rosa Brillante</option>
                        <option value="3">Pálidas</option>
                        <option value="4">Azuladas</option>
                        <option value="5">Rojo Peligro</option>
                        <option value="6">Cianóticas</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Llenado Capilar</label>
                    <select name="capillary_refill_time" required>
                        <option value="1">< 3 seg</option>
                        <option value="2">> 3 seg</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Nivel de Dolor</label>
                    <select name="pain" required>
                        <option value="1">Alerta/Sin dolor</option>
                        <option value="2">Deprimido</option>
                        <option value="3">Intermitente Leve</option>
                        <option value="4">Intermitente Grave</option>
                        <option value="5">Continuo Grave</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Peristaltismo</label>
                    <select name="peristalsis" required>
                        <option value="1">Hipermóvil</option>
                        <option value="2">Normal</option>
                        <option value="3">Hipomóvil</option>
                        <option value="4">Ausente</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Distensión Abdominal</label>
                    <select name="abdominal_distension" required>
                        <option value="1">Ninguna</option>
                        <option value="2">Leve</option>
                        <option value="3">Moderada</option>
                        <option value="4">Grave</option>
                    </select>
                </div>
                <!-- Laboratorio -->
                <div class="input-group">
                    <label>PCV (%)</label>
                    <input type="number" step="0.1" name="packed_cell_volume" value="45" required>
                </div>
                <div class="input-group">
                    <label>Proteína Total (g/dL)</label>
                    <input type="number" step="0.1" name="total_protein" value="7.5" required>
                </div>

                <button type="submit" class="btn-predict">Realizar Predicción</button>
            </form>

            <div class="result-section" id="resultCard">
                <div id="loading"><div class="spinner"></div></div>
                <div id="placeholder">
                    <p style="color: #64748b;">Ingresa los datos y haz clic en predecir</p>
                </div>
                
                <div id="resultContent" style="display: none;">
                    <div id="badge" class="result-badge">---</div>
                    <div class="prob-text" id="probValue">0%</div>
                    <div class="prob-label">Confianza del Modelo</div>
                    <div class="prob-container">
                        <div class="prob-bar-bg">
                            <div id="probFill" class="prob-bar-fill"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="status-footer">
            <span id="statusDot" class="success-dot">●</span> 
            Status del Sistema: <span id="statusLabel">Cargando...</span>
        </div>
    </div>

    <script>
        const statusLabel = document.getElementById('statusLabel');
        const statusDot = document.getElementById('statusDot');
        const form = document.getElementById('predictForm');
        const loading = document.getElementById('loading');
        const placeholder = document.getElementById('placeholder');
        const resultContent = document.getElementById('resultContent');
        const badge = document.getElementById('badge');
        const probValue = document.getElementById('probValue');
        const probFill = document.getElementById('probFill');

        // Check overall status
        fetch('/status').then(res => res.json()).then(data => {
            statusLabel.innerText = data.status_modelo;
            if (data.status_modelo.includes('éxito')) {
                statusDot.className = 'success-dot';
            } else {
                statusDot.className = 'error-dot';
            }
        }).catch(() => {
            statusLabel.innerText = "Error de conexión con el servidor";
            statusDot.className = 'error-dot';
        });

        form.onsubmit = async (e) => {
            e.preventDefault();
            
            // UI States
            placeholder.style.display = 'none';
            resultContent.style.display = 'none';
            loading.style.display = 'block';

            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            try {
                const response = await fetch('/predecir', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                // Update UI
                loading.style.display = 'none';
                resultContent.style.display = 'block';
                
                // Badge
                badge.innerText = result.resultado_clinico;
                badge.style.display = 'inline-block';
                if (result.prediccion === 1) {
                    badge.className = 'result-badge requires-surgery';
                } else {
                    badge.className = 'result-badge medical-treatment';
                }

                // Probabilities
                const prob = result.probabilidad_cirugia;
                probValue.innerText = prob + '%';
                probFill.style.width = prob + '%';

            } catch (err) {
                alert("Error al realizar la predicción");
                loading.style.display = 'none';
                placeholder.style.display = 'block';
            }
        };
    </script>
</body>
</html>
"""

# --- RUTAS ---

@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/status")
def get_status():
    return {
        "status_modelo": status_modelo,
        "api": "Predictor de Cólico Equino v1.0"
    }

@app.post("/predecir")
def predecir_lesion(datos: DatosCaballo):
    df = pd.DataFrame([datos.dict()])
    
    try:
        prediccion = modelo.predict(df)[0]
        probabilidades = modelo.predict_proba(df)[0]
        
        resultado = "Sí requiere cirugía (1)" if prediccion == 1 else "Tratamiento Médico (2)"
        
        # Ajuste de probabilidad para cirugía
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)