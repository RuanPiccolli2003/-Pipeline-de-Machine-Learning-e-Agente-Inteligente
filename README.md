# Detecção de Fraudes em Cartões de Crédito com Agente de IA

Este repositório contém a solução final do projeto de detecção de fraudes em transações de cartão de crédito. A solução inclui um pipeline completo de Machine Learning para treinar e avaliar diferentes modelos preditivos, além de uma aplicação web com um Agente Especialista (LLM) capaz de explicar os resultados das predições de forma clara e objetiva para o usuário final.

## 🛠️ Como rodar o projeto

### 1. Requisitos
Certifique-se de ter o Python 3.8+ instalado. Instale as dependências necessárias (recomendamos o uso de um ambiente virtual):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib flask huggingface_hub
```

### 2. Base de Dados
O projeto requer o arquivo `credit_card_transactions.csv` na raiz desta pasta `FINAL/`. Como se trata de um arquivo grande, certifique-se de que ele esteja presente antes de rodar os scripts.

### 3. Executando o Pipeline de Machine Learning
Para realizar o pré-processamento, gerar gráficos exploratórios e treinar os 4 algoritmos (Regressão Logística, KNN, MLP, Naive Bayes), execute o script principal de treinamento:

```bash
python backup2.py
```
*(Nota: O script de treinamento foi fornecido como `backup2.py`. Ele irá gerar os gráficos em `.png` e exportar o `modelo_final.joblib`)*.

### 4. Iniciando a Aplicação Web (Agente LLM)
Após a geração do modelo, inicie o servidor Flask:

```bash
python app2.py
```
Acesse a aplicação pelo navegador no endereço: `http://localhost:5000`

---

## 📅 Diário de Bordo de Contribuições

Durante os 15 dias de desenvolvimento do projeto, os integrantes da equipe colaboraram nas seguintes atividades. *(Por favor, preencham com os nomes e descrições detalhadas do que cada integrante fez)*.

### Integrante 1: [Fernando Gonçalves]
* **Dia 1-5:** [Exploração inicial dos dados e limpeza.]
* **Dia 6-10:** [Implementação do sub-sampling e treinamento dos modelos de ML.]
* **Dia 11-15:** [revisão do relatório técnico.]

### Integrante 2: [Ruan Carlos]
* **Dia 1-5:** [Pesquisa e configuração do ambiente Flask e templates HTML.]
* **Dia 6-10:** [Integração com a API do HuggingFace para o Agente LLM.]
* **Dia 11-15:** [Ajustes na interface web e testes de ponta a ponta da aplicação.]

