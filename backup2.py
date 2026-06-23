import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.calibration import CalibratedClassifierCV

from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    print(f"[INFO] Dataset original carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    
    fraudes = df[df["is_fraud"] == 1]
    nao_fraudes = df[df["is_fraud"] == 0]
    
    n_fraudes = len(fraudes)
    
    n_nao_fraudes = n_fraudes * 9
    
    MAX_FRAUDES = 10000
    if n_fraudes > MAX_FRAUDES:
        fraudes = fraudes.sample(n=MAX_FRAUDES, random_state=42)
        n_fraudes = MAX_FRAUDES
        n_nao_fraudes = MAX_FRAUDES * 9
        
    if len(nao_fraudes) > n_nao_fraudes:
        nao_fraudes = nao_fraudes.sample(n=n_nao_fraudes, random_state=42)
        
    df_reduzido = pd.concat([fraudes, nao_fraudes]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[INFO] Dataset amostrado (90% Não Fraude, 10% Fraude): {df_reduzido.shape[0]} linhas")
    return df_reduzido

def traduzir_colunas(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {
        "trans_date_trans_time": "data_hora_transacao", "cc_num": "num_cartao",
        "merchant": "comerciante", "category": "categoria", "amt": "valor",
        "first": "nome", "last": "sobrenome", "gender": "genero",
        "street": "endereco", "city": "cidade", "state": "estado",
        "zip": "cep", "lat": "latitude", "long": "longitude",
        "city_pop": "populacao_cidade", "job": "profissao", "dob": "data_nascimento",
        "trans_num": "id_transacao", "unix_time": "unix_hora",
        "merch_lat": "latitude_comerciante", "merch_long": "longitude_comerciante",
        "is_fraud": "eh_fraude", "merch_zipcode": "cep_comerciante",
    }
    return df.rename(columns=mapa)

def converter_datas(df: pd.DataFrame) -> pd.DataFrame:
    if "data_hora_transacao" in df.columns:
        df["data_hora_transacao"] = pd.to_datetime(df["data_hora_transacao"], errors="coerce")
        df["hora_transacao"] = df["data_hora_transacao"].dt.hour
        df["dia_semana"] = df["data_hora_transacao"].dt.dayofweek
    if "data_nascimento" in df.columns:
        df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce")
        ref_date = df["data_hora_transacao"].min() if "data_hora_transacao" in df.columns else pd.Timestamp("2019-01-01")
        df["idade"] = (ref_date - df["data_nascimento"]).dt.days // 365
    return df

def gerar_graficos_exploratorios(df: pd.DataFrame) -> None:
    print("\n[INFO] Gerando gráficos exploratórios...")

    colunas_numericas = [
        "valor", "latitude", "longitude", "populacao_cidade",
        "latitude_comerciante", "longitude_comerciante",
        "hora_transacao", "dia_semana", "idade", "eh_fraude"
    ]
    cols_disponiveis = [c for c in colunas_numericas if c in df.columns]
    corr = df[cols_disponiveis].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Mapa de Correlação das Variáveis Numéricas", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("grafico_correlacao.png", bbox_inches="tight")
    plt.close()
    print("  [OK] grafico_correlacao.png salvo")

    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df.copy()
    df_plot["Classe"] = df_plot["eh_fraude"].map({0: "Não Fraude", 1: "Fraude"})
    sns.boxplot(
        data=df_plot, x="Classe", y="valor", palette={"Não Fraude": "#3498db", "Fraude": "#e74c3c"},
        showfliers=True, ax=ax
    )
    ax.set_title("Distribuição do Valor por Classe", fontsize=15, fontweight="bold")
    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Valor da Transação (R$)", fontsize=12)
    plt.tight_layout()
    plt.savefig("grafico_boxplot.png", bbox_inches="tight")
    plt.close()
    print("  [OK] grafico_boxplot.png salvo")

    if "categoria" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 7))
        ordem = df.groupby("categoria")["eh_fraude"].sum().sort_values(ascending=False).index
        sns.countplot(
            data=df[df["eh_fraude"] == 1], x="categoria", order=ordem,
            palette="Reds_r", ax=ax
        )
        ax.set_title("Frequência de Fraudes por Categoria", fontsize=15, fontweight="bold")
        ax.set_xlabel("Categoria", fontsize=12)
        ax.set_ylabel("Quantidade de Fraudes", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("grafico_frequencia.png", bbox_inches="tight")
        plt.close()
        print("  [OK] grafico_frequencia.png salvo")

    print("[INFO] Gráficos exploratórios gerados com sucesso!\n")

def selecionar_features(df: pd.DataFrame) -> tuple:
    colunas_features = [
        "valor", "latitude", "longitude", "populacao_cidade",
        "latitude_comerciante", "longitude_comerciante",
        "hora_transacao", "dia_semana", "idade"
    ]

    le = None
    if "genero" in df.columns:
        df["genero_cod"] = (df["genero"] == "M").astype(int)
        colunas_features.append("genero_cod")
    if "categoria" in df.columns:
        le = LabelEncoder()
        df["categoria_cod"] = le.fit_transform(df["categoria"].astype(str))
        colunas_features.append("categoria_cod")

    colunas_disponiveis = [c for c in colunas_features if c in df.columns]
    X = df[colunas_disponiveis].copy().fillna(0)
    y = df["eh_fraude"].copy()
    return X, y, le, colunas_disponiveis

def dividir_e_balancear_dados(X, y) -> tuple:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    rus = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    
    print("\n" + "=" * 60)
    print(" DISTRIBUIÇÃO DAS CLASSES (TREINO APÓS UNDERSAMPLING)")
    print("=" * 60)
    print(y_train_bal.value_counts())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)  
    
    return X_train_scaled, X_test_scaled, y_train_bal, y_test, scaler

def calcular_especificidade(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0

def avaliar_modelo(nome: str, y_train_true, y_train_pred, y_test_true, y_test_pred) -> dict:
    acc_test = accuracy_score(y_test_true, y_test_pred)
    prec_test = precision_score(y_test_true, y_test_pred, zero_division=0)
    rec_test = recall_score(y_test_true, y_test_pred, zero_division=0)  
    f1_test = f1_score(y_test_true, y_test_pred, zero_division=0)
    espec_test = calcular_especificidade(y_test_true, y_test_pred)

    f1_train = f1_score(y_train_true, y_train_pred, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f" MODELO: {nome}")
    print(f"{'=' * 60}")
    print(f" Acurácia Teste:          {acc_test:.4f}")
    print(f" Precisão Teste:          {prec_test:.4f}")
    print(f" Sensibilidade Teste:     {rec_test:.4f}")
    print(f" Especificidade Teste:    {espec_test:.4f}")
    print(f" F1-Score Teste:          {f1_test:.4f}")
    print(f" F1-Score Treino:         {f1_train:.4f}")
    print(f"\n[TESTE] Relatório de Classificação Final:")
    print(classification_report(y_test_true, y_test_pred, zero_division=0))

    return {
        "modelo": nome,
        "acuracia": acc_test,
        "precisao": prec_test,
        "sensibilidade": rec_test,
        "especificidade": espec_test,
        "f1_test": f1_test,
        "f1_train": f1_train,
    }

def plotar_matriz_confusao(nome: str, y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Não Fraude", "Fraude"], cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de Confusão - {nome} (Teste)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"matriz_{nome.lower().replace(' ', '_')}.png", bbox_inches="tight")
    plt.close()

def treinar_regressao_logistica(X_train, y_train, X_test, y_test) -> tuple:
    
    modelo_base = LogisticRegression(class_weight='balanced', C=0.1, max_iter=2000, random_state=42)
    
    modelo = CalibratedClassifierCV(estimator=modelo_base, method='sigmoid', cv=5)
    
    modelo.fit(X_train, y_train)
    
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    resultados = avaliar_modelo("Regressão Logística", y_train, y_train_pred, y_test, y_test_pred)
    plotar_matriz_confusao("Regressão Logística", y_test, y_test_pred)
    return resultados, modelo

def treinar_knn(X_train, y_train, X_test, y_test) -> tuple:
    modelo_base = KNeighborsClassifier(n_neighbors=51, weights='uniform')
    modelo = CalibratedClassifierCV(estimator=modelo_base, method='sigmoid', cv=5)
    modelo.fit(X_train, y_train)
    
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    resultados = avaliar_modelo("KNN", y_train, y_train_pred, y_test, y_test_pred)
    plotar_matriz_confusao("KNN", y_test, y_test_pred)
    return resultados, modelo

def treinar_mlp(X_train, y_train, X_test, y_test) -> tuple:
    modelo_base = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.01, max_iter=500, random_state=42, early_stopping=True)
    modelo = CalibratedClassifierCV(estimator=modelo_base, method='sigmoid', cv=5)
    modelo.fit(X_train, y_train)
    
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    resultados = avaliar_modelo("MLP", y_train, y_train_pred, y_test, y_test_pred)
    plotar_matriz_confusao("MLP", y_test, y_test_pred)
    return resultados, modelo

def treinar_naive_bayes(X_train, y_train, X_test, y_test) -> tuple:
    modelo_base = GaussianNB()
    modelo = CalibratedClassifierCV(estimator=modelo_base, method='sigmoid', cv=5)
    modelo.fit(X_train, y_train)
    
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)
    
    resultados = avaliar_modelo("Naive Bayes", y_train, y_train_pred, y_test, y_test_pred)
    plotar_matriz_confusao("Naive Bayes", y_test, y_test_pred)
    return resultados, modelo

def exportar_melhor_modelo(resultados_lista, modelos_dict, scaler, label_encoder, features_list):
    df_comp = pd.DataFrame(resultados_lista)
    
    idx_melhor = df_comp["f1_test"].idxmax()
    nome_melhor = df_comp.loc[idx_melhor, "modelo"]
    f1_melhor = df_comp.loc[idx_melhor, "f1_test"]
    
    print(f"\n{'=' * 60}")
    print(f" [WINNER] MELHOR MODELO: {nome_melhor} (F1-Score: {f1_melhor:.4f})")
    print(f"{'=' * 60}")
    
    pacote = {
        "modelo": modelos_dict[nome_melhor],
        "nome_modelo": nome_melhor,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": features_list,
        "metricas": df_comp.loc[idx_melhor].to_dict(),
    }
    
    caminho_saida = "modelo_final.joblib"
    joblib.dump(pacote, caminho_saida)
    print(f"[INFO] Modelo exportado para: {caminho_saida}")
    print(f"[INFO] Conteúdo do pacote: modelo, scaler, label_encoder, features, métricas")
    
    return nome_melhor

def main():
    CAMINHO_CSV = "credit_card_transactions.csv"
    
    print("[INFO] Iniciando pipeline...")
    df = carregar_dados(CAMINHO_CSV)
    df = traduzir_colunas(df)
    
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    if "id" in df.columns: df = df.drop(columns=["id"])
    if "cep_comerciante" in df.columns: df["cep_comerciante"] = df["cep_comerciante"].fillna(0).astype(int)
    
    df = converter_datas(df)
    
    df = df.drop_duplicates()

    gerar_graficos_exploratorios(df)

    X, y, label_encoder, features_list = selecionar_features(df)
    X_train_bal, X_test, y_train_bal, y_test, scaler = dividir_e_balancear_dados(X, y)

    resultados = []
    modelos = {}

    res, mod = treinar_regressao_logistica(X_train_bal, y_train_bal, X_test, y_test)
    resultados.append(res); modelos[res["modelo"]] = mod

    res, mod = treinar_knn(X_train_bal, y_train_bal, X_test, y_test)
    resultados.append(res); modelos[res["modelo"]] = mod

    res, mod = treinar_mlp(X_train_bal, y_train_bal, X_test, y_test)
    resultados.append(res); modelos[res["modelo"]] = mod

    res, mod = treinar_naive_bayes(X_train_bal, y_train_bal, X_test, y_test)
    resultados.append(res); modelos[res["modelo"]] = mod

    df_comp = pd.DataFrame(resultados)
    print("\n" + "=" * 60)
    print(" RESUMO FINAL DAS MÉTRICAS (4 ALGORITMOS)")
    print("=" * 60)
    print(df_comp.to_string(index=False))

    exportar_melhor_modelo(resultados, modelos, scaler, label_encoder, features_list)

if __name__ == "__main__":
    main()
