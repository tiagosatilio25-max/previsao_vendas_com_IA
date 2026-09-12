"""
Aplicação Web de Previsão de Vendas com Streamlit e TensorFlow.
Autor: Especialista ML & Python
Descrição: Interface interativa para análise exploratória de vendas e
            previsão de séries temporais via Rede Neural Keras/TensorFlow.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuração da página Streamlit
st.set_page_config(
    page_title="Previsão de Vendas com TensorFlow",
    page_icon="📈",
    layout="wide"
)


@st.cache_data
def carregar_dados_padrao() -> pd.DataFrame:
    """Carrega dataset inicial via dicionário Python."""
    dados_vendas = {
        "Data": pd.date_range(start="2026-01-01", periods=12, freq="M"),
        "Vendas_Unidades": [120, 135, 150, 160, 190, 210, 230, 250, 280, 300, 310, 340],
        "Investimento_Mkt": [10, 12, 15, 14, 18, 20, 22, 25, 27, 30, 31, 35]
    }
    return pd.DataFrame(dados_vendas)


def preparar_dados(vendas: np.ndarray, janela: int):
    """Prepara as sequências de entrada (X) e alvo (y) para o modelo."""
    X, y = [], []
    for i in range(len(vendas) - janela):
        X.append(vendas[i : i + janela])
        y.append(vendas[i + janela])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def treinar_modelo_tensorflow(X: np.ndarray, y: np.ndarray, epocas: int) -> tf.keras.Model:
    """Compila e treina uma rede neural de regressão."""
    modelo = Sequential([
        Dense(16, activation='relu', input_shape=(X.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    modelo.fit(X, y, epochs=epocas, verbose=0)
    return modelo


def main():
    st.title("📈 Dashboard de Previsão de Vendas com IA")
    st.write("Aplicação interativa para análise exploratória e previsão via TensorFlow.")

    # Painel Lateral para Configurações
    st.sidebar.header("⚙️ Configurações do Modelo")
    tamanho_janela = st.sidebar.slider("Janela de Histórico (meses)", min_value=2, max_value=6, value=3)
    epocas_treino = st.sidebar.slider("Épocas de Treinamento", min_value=100, max_value=1000, value=500, step=100)

    # Carregamento de dados
    df = carregar_dados_padrao()

    # Layout em Abas
    aba1, aba2, aba3 = st.tabs(["📊 Dados & Estatísticas", "🧠 Treinamento & IA", "🔮 Previsão"])

    with aba1:
        st.subheader("Visualização e Análise Exploratória")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Tabela de Dados**")
            st.dataframe(df, use_container_width=True)
            
        with col2:
            st.markdown("**Estatísticas Descritivas**")
            st.dataframe(df.describe().T, use_container_width=True)
            st.line_chart(df.set_index("Data")[["Vendas_Unidades", "Investimento_Mkt"]])

    vendas_array = df["Vendas_Unidades"].values.astype(np.float32)

    with aba2:
        st.subheader("Processamento e Treinamento do TensorFlow")
        
        if len(vendas_array) <= tamanho_janela:
            st.error("O número de registros precisa ser maior que o tamanho da janela.")
            return

        X, y = preparar_dados(vendas_array, janela=tamanho_janela)

        st.write(f"**Tamanho das amostras criadas (X):** `{X.shape}`")
        st.write(f"**Tamanho dos alvos (y):** `{y.shape}`")

        if st.button("🚀 Treinar Rede Neural"):
            with st.spinner("Treinando modelo TensorFlow..."):
                st.session_state["modelo"] = treinar_modelo_tensorflow(X, y, epocas_treino)
                st.session_state["treinado"] = True
            st.success("Modelo treinado com sucesso!")

    with aba3:
        st.subheader("Gerar Projeção de Vendas")
        
        if st.session_state.get("treinado", False):
            modelo = st.session_state["modelo"]
            ultimos_valores = vendas_array[-tamanho_janela:]
            
            entrada = np.array([ultimos_valores], dtype=np.float32)
            predicao = modelo.predict(entrada, verbose=0)[0][0]

            col_metrica1, col_metrica2 = st.columns(2)
            with col_metrica1:
                st.metric(
                    label=f"Última venda registrada ({tamanho_janela}º mês)", 
                    value=f"{ultimos_valores[-1]:.0f} un."
                )
            with col_metrica2:
                st.metric(
                    label="Previsão para o Próximo Mês", 
                    value=f"{predicao:.2f} un.", 
                    delta=f"{predicao - ultimos_valores[-1]:.2f} un."
                )

            # Histórico + Projeção
            df_projecao = df.copy()
            proximo_mes = df_projecao["Data"].iloc[-1] + pd.DateOffset(months=1)
            
            novo_registro = pd.DataFrame({
                "Data": [proximo_mes],
                "Vendas_Unidades": [predicao],
                "Investimento_Mkt": [np.nan]
            })
            
            df_grafico = pd.concat([df_projecao, novo_registro], ignore_index=True)
            st.subheader("Tendência Histórica + Projeção")
            st.line_chart(df_grafico.set_index("Data")["Vendas_Unidades"])
        else:
            st.info("Treine o modelo na aba **🧠 Treinamento & IA** para habilitar as previsões.")


if __name__ == "__main__":
    main()