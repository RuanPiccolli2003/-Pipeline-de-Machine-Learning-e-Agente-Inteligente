import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

MODELO_PATH = "modelo_final.joblib"
pacote = None

def carregar_modelo():
    global pacote
    if os.path.exists(MODELO_PATH):
        pacote = joblib.load(MODELO_PATH)
        print(f"[INFO] Modelo carregado: {pacote['nome_modelo']}")
        print(f"[INFO] Features: {pacote['features']}")
        print(f"[INFO] Métricas: {pacote['metricas']}")
    else:
        print(f"[AVISO] Modelo não encontrado em '{MODELO_PATH}'. Execute analise_fraude2.py primeiro.")

CATEGORIAS = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets",
    "misc_net", "misc_pos", "personal_care", "shopping_net",
    "shopping_pos", "travel"
]

SYSTEM_PROMPT = """Você é um Agente Especialista em Detecção de Fraudes Financeiras. 
Sua função é analisar o resultado de um modelo de Machine Learning que prevê fraudes 
em transações de cartão de crédito e explicar o resultado ao usuário final de forma 
clara, fundamentada e profissional.

REGRAS IMPORTANTES:
1. NUNCA invente dados ou informações que não foram fornecidos
2. Baseie sua análise EXCLUSIVAMENTE nos dados da transação e no resultado do modelo
3. Explique os fatores que podem ter contribuído para a classificação
4. Use linguagem acessível, evitando jargões técnicos excessivos
5. Seja objetivo e direto na explicação
6. Mencione a confiança do modelo (probabilidade) na sua explicação
7. Se for fraude, sugira ações preventivas genéricas
8. Se for legítima, explique brevemente por que a transação parece normal
9. Responda SEMPRE em português brasileiro
10. Mantenha a resposta entre 3 e 6 parágrafos"""

def obter_explicacao_agente(dados_transacao: dict, predicao: int, probabilidade: float) -> str:
    hf_token = "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"
    
    label = "FRAUDE DETECTADA" if predicao == 1 else "TRANSAÇÃO LEGÍTIMA"
    
    prompt_usuario = f"""Analise a seguinte transação de cartão de crédito e explique o resultado da predição do modelo:

DADOS DA TRANSAÇÃO:
- Valor: R$ {dados_transacao.get('valor', 0):.2f}
- Categoria: {dados_transacao.get('categoria', 'N/A')}
- Gênero do titular: {"Masculino" if dados_transacao.get('genero', 'M') == 'M' else "Feminino"}
- Localização do cliente: ({dados_transacao.get('latitude', 0):.4f}, {dados_transacao.get('longitude', 0):.4f})
- População da cidade: {dados_transacao.get('populacao_cidade', 0):,}
- Localização do comerciante: ({dados_transacao.get('latitude_comerciante', 0):.4f}, {dados_transacao.get('longitude_comerciante', 0):.4f})
- Hora da transação: {dados_transacao.get('hora_transacao', 0)}h
- Dia da semana: {['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'][int(dados_transacao.get('dia_semana', 0))]}
- Idade do titular: {dados_transacao.get('idade', 0)} anos

RESULTADO DO MODELO DE ML:
- Classificação: {label}
- Probabilidade de fraude: {probabilidade * 100:.1f}%
- Modelo utilizado: {pacote['nome_modelo'] if pacote else 'N/A'}

Por favor, explique este resultado ao usuário de forma clara e fundamentada."""

    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token if hf_token else None,
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario},
        ]
        
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=messages,
            max_tokens=800,
            temperature=0.4,
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERRO] Falha na API do Hugging Face: {e}")
        
        return gerar_explicacao_fallback(dados_transacao, predicao, probabilidade)

def gerar_explicacao_fallback(dados_transacao: dict, predicao: int, probabilidade: float) -> str:
    valor = dados_transacao.get('valor', 0)
    hora = dados_transacao.get('hora_transacao', 0)
    categoria = dados_transacao.get('categoria', 'N/A')
    
    if predicao == 1:
        fatores = []
        if valor > 500:
            fatores.append(f"o valor elevado da transação (R$ {valor:.2f})")
        if hora >= 23 or hora <= 5:
            fatores.append(f"o horário incomum ({hora}h)")
        if not fatores:
            fatores.append("padrões identificados pelo modelo nos dados da transação")
        
        fatores_texto = ", ".join(fatores)
        return (
            f"O modelo classificou esta transação como POTENCIAL FRAUDE com "
            f"{probabilidade * 100:.1f}% de confiança.\n\n"
            f"Os possíveis fatores que contribuíram para essa classificação incluem: "
            f"{fatores_texto}.\n\n"
            f"Recomenda-se verificar a transação junto ao titular do cartão e, "
            f"se confirmada a fraude, bloquear o cartão imediatamente."
        )
    else:
        return (
            f"O modelo classificou esta transação como LEGÍTIMA com "
            f"{(1 - probabilidade) * 100:.1f}% de confiança.\n\n"
            f"A transação de R$ {valor:.2f} na categoria '{categoria}' "
            f"apresenta características dentro dos padrões normais de uso, "
            f"sem indicadores significativos de atividade fraudulenta."
        )

@app.route("/")
def index():
    return render_template("index.html", categorias=CATEGORIAS)

@app.route("/predict", methods=["POST"])
def predict():
    if pacote is None:
        return jsonify({"erro": "Modelo não carregado. Execute analise_fraude2.py primeiro."}), 500
    
    try:
        dados = request.get_json()
        
        modelo = pacote["modelo"]
        scaler = pacote["scaler"]
        le = pacote["label_encoder"]
        features = pacote["features"]
        
        vetor = []
        for feat in features:
            if feat == "genero_cod":
                vetor.append(1 if dados.get("genero", "M") == "M" else 0)
            elif feat == "categoria_cod":
                cat = dados.get("categoria", "misc_net")
                if le is not None and cat in le.classes_:
                    vetor.append(le.transform([cat])[0])
                else:
                    vetor.append(0)
            else:
                val = dados.get(feat, 0)
                try:
                    vetor.append(float(val))
                except (ValueError, TypeError):
                    vetor.append(0.0)
        
        X_input = np.array([vetor])
        X_scaled = scaler.transform(X_input)
        
        predicao = int(modelo.predict(X_scaled)[0])
        
        if hasattr(modelo, "predict_proba"):
            prob = float(modelo.predict_proba(X_scaled)[0][1])
        else:
            prob = float(predicao)
        
        label = "FRAUDE DETECTADA" if predicao == 1 else "TRANSAÇÃO LEGÍTIMA"
        
        explicacao = obter_explicacao_agente(dados, predicao, prob)
        
        return jsonify({
            "predicao": predicao,
            "probabilidade": round(prob, 4),
            "label": label,
            "modelo_usado": pacote["nome_modelo"],
            "explicacao_agente": explicacao,
        })
        
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

@app.route("/info", methods=["GET"])
def info():
    if pacote is None:
        return jsonify({"erro": "Modelo não carregado."})
    
    return jsonify({
        "nome_modelo": pacote["nome_modelo"],
        "features": pacote["features"],
        "metricas": {k: round(v, 4) if isinstance(v, float) else v for k, v in pacote["metricas"].items()},
    })

if __name__ == "__main__":
    carregar_modelo()
    print("\n" + "=" * 60)
    print(" [INIT] Servidor Flask iniciado!")
    print(" [URL] Acesse: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
