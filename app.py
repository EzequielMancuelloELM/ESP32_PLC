import os, json, re, requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY", "")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
ALLOWED_TIMERS = {0, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000}

SYSTEM_PROMPT = r'''Sos un ingeniero de automatización que convierte una descripción de proceso en una PROPUESTA de lógica para un PLC educativo ESP32.
Respondé exclusivamente con un objeto JSON válido, sin Markdown ni texto externo, con estas claves:
needs_clarification (boolean), summary (string), questions (array de strings), warnings (array de strings), inputs (array de {id integer, name string}), outputs (array de {id integer, name string, conditions array de {source string E o S, id integer, active boolean, operator string AND u OR, timer_ms integer} }).
Reglas obligatorias:
- No inventes sensores, actuadores, cantidades ni estados no mencionados.
- Si falta información, needs_clarification=true, hacé preguntas concretas y dejá inputs y outputs vacíos.
- Máximo 8 entradas E1..E8, 8 salidas S1..S8, 8 condiciones por salida. IDs consecutivos desde 1.
- source E referencia entrada; source S referencia salida. active=true significa señal activa; false, inactiva.
- operator es la conexión con la condición anterior; primera condición siempre AND.
- La lógica del editor representa OR como ramas paralelas y AND como serie dentro de cada rama. Expresá la lógica como suma de productos (grupos AND unidos por OR). Si la lógica mixta no queda clara, pedí aclaración.
- timer_ms es TON de ESA condición. Valores permitidos: 0,250,500,1000,2000,5000,10000,30000,60000. Usá 0 si no se pide retardo.
- Si se necesita memoria SET/RESET, enclavamiento, contador, parada de emergencia funcional o lógica no soportada, no la simules: pedí aclaración y advertí que no está soportada.
- Configurá solo las señales necesarias. No asignes GPIO; se asignan después en el editor.
- Si el proceso involucra motores, bombas, calor, presión, movimiento o emergencia, agregá advertencia de seguridad. La parada de emergencia debe ser independiente del software.
- Este formato define cuándo una salida está activa; no implementa memoria de marcha. Si el requisito depende de memoria de estado, pedí aclaración.
'''

def validate_proposal(data):
    if not isinstance(data, dict): raise ValueError("La respuesta de IA no es un objeto JSON")
    required = ["needs_clarification", "summary", "questions", "warnings", "inputs", "outputs"]
    if any(k not in data for k in required): raise ValueError("La respuesta no contiene todos los campos requeridos")
    if data["needs_clarification"]:
        data["inputs"], data["outputs"] = [], []
        return data
    ins, outs = data["inputs"], data["outputs"]
    if not 1 <= len(ins) <= 8 or not 1 <= len(outs) <= 8: raise ValueError("La propuesta debe incluir entre 1 y 8 entradas y salidas")
    in_ids, out_ids = [x.get("id") for x in ins], [x.get("id") for x in outs]
    if sorted(set(in_ids)) != list(range(1, max(in_ids)+1)) or max(in_ids)>8: raise ValueError("Los IDs de entradas deben ser consecutivos desde E1")
    if sorted(set(out_ids)) != list(range(1, max(out_ids)+1)) or max(out_ids)>8: raise ValueError("Los IDs de salidas deben ser consecutivos desde S1")
    for out in outs:
        conds = out.get("conditions", [])
        if not 1 <= len(conds) <= 8: raise ValueError(f"S{out['id']} debe tener entre 1 y 8 condiciones")
        for i,c in enumerate(conds):
            if c.get("source") not in ("E","S"): raise ValueError("Tipo de señal inválido")
            if c["source"] == "E" and c.get("id") not in set(in_ids): raise ValueError("Condición refiere a una entrada no declarada")
            if c["source"] == "S" and c.get("id") not in set(out_ids): raise ValueError("Condición refiere a una salida no declarada")
            if c.get("timer_ms") not in ALLOWED_TIMERS: raise ValueError("Temporizador no permitido")
            if c.get("operator") not in ("AND","OR"): raise ValueError("Operador lógico inválido")
            if i == 0: c["operator"] = "AND"
    return data

@app.get("/health")
def health():
    return jsonify({"ok": True, "provider": "Google Gemini API free tier (según disponibilidad y límites vigentes)", "model": MODEL, "api_key_configured": bool(API_KEY)})

@app.post("/interpret")
def interpret():
    if not API_KEY: return jsonify({"error":"Falta configurar GEMINI_API_KEY en el servidor"}), 503
    body = request.get_json(silent=True) or {}
    description = str(body.get("description", "")).strip()
    if len(description) < 12: return jsonify({"error":"La descripción es demasiado breve"}), 400
    if len(description) > 5000: return jsonify({"error":"La descripción supera los 5000 caracteres"}), 400
    payload = {"system_instruction":{"parts":[{"text":SYSTEM_PROMPT}]},"contents":[{"role":"user","parts":[{"text":"Convertí esta descripción en una propuesta. No actives salidas; solo devolvé la configuración para revisión.\n\nDESCRIPCIÓN:\n"+description}]}],"generationConfig":{"temperature":0.1,"responseMimeType":"application/json"}}
    try:
        r = requests.post(API_URL, params={"key":API_KEY}, json=payload, timeout=90)
        if r.status_code != 200:
            app.logger.warning("Gemini status %s: %s", r.status_code, r.text[:1000])
            return jsonify({"error":f"Gemini respondió HTTP {r.status_code}. Revisá la clave, el modelo y los límites gratuitos."}), 502
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        data = validate_proposal(json.loads(raw))
        return jsonify(data)
    except requests.Timeout:
        return jsonify({"error":"La IA tardó demasiado. Probá nuevamente en unos segundos."}), 504
    except Exception as exc:
        app.logger.exception("Fallo interpretando descripción")
        return jsonify({"error":"No se pudo generar una propuesta válida. Probá reformular la descripción o revisar los logs del servicio."}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
