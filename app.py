# app.py

import streamlit as st
import pandas as pd
import os
from bot_core_v53 import CONFIG, estatisticas, iniciar_bot_em_thread

st.set_page_config(page_title="WebBot IA v53 - PyTorch Edition", layout="wide")
st.title("🧠 WebBot IA v53 - Controle com PyTorch + Ensemble + Contexto")

# ========== Sidebar: Configurações ==========
st.sidebar.header("⚙️ Configurações do Bot")

CONFIG["STAKE_INICIAL"] = st.sidebar.number_input("Stake Inicial", 0.1, 100.0, value=1.0)
CONFIG["STAKE_MAX"] = st.sidebar.number_input("Stake Máxima", 1.0, 1000.0, value=100.0)
CONFIG["MARTINGALE_MULT"] = st.sidebar.slider("Multiplicador Martingale", 1.0, 10.0, value=2.0, step=0.1)
CONFIG["DURATION"] = st.sidebar.slider("Duração da Operação (ticks)", 1, 10, value=2)
CONFIG["LIMIAR_CONF"] = st.sidebar.slider("Confiança Mínima da IA", 0.5, 1.0, value=0.7, step=0.01)

if st.sidebar.button("🚀 Iniciar Bot IA"):
    iniciar_bot_em_thread()
    st.sidebar.success("Bot iniciado! Rodando em segundo plano...")

# ========== Upload / Download ==========
st.sidebar.markdown("---")
st.sidebar.header("📁 Arquivos")

with st.sidebar.expander("📤 Baixar CSVs"):
    if os.path.exists("diagnostico_ia.csv"):
        with open("diagnostico_ia.csv", "rb") as f:
            st.download_button("Diagnóstico IA", f, file_name="diagnostico_ia.csv")
    if os.path.exists("dados_treinamento.csv"):
        with open("dados_treinamento.csv", "rb") as f:
            st.download_button("Dados de Treinamento", f, file_name="dados_treinamento.csv")

with st.sidebar.expander("📥 Enviar CSVs"):
    file_diag = st.file_uploader("Substituir Diagnóstico", type="csv")
    if file_diag:
        with open("diagnostico_ia.csv", "wb") as f:
            f.write(file_diag.getbuffer())
        st.success("Diagnóstico atualizado.")

    file_dados = st.file_uploader("Substituir Dados de Treinamento", type="csv")
    if file_dados:
        with open("dados_treinamento.csv", "wb") as f:
            f.write(file_dados.getbuffer())
        st.success("Dados de treino atualizados.")

# ========== Painel Principal ==========
st.subheader("📊 Estatísticas da IA")
col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Acertos", estatisticas["acertos"])
col2.metric("❌ Erros", estatisticas["erros"])
col3.metric("🛑 Bloqueios", estatisticas["bloqueios"])
col4.metric("💰 Lucro Total", f"${estatisticas['lucro_total']:.2f}")

# ========== Diagnóstico ==========
st.subheader("📋 Diagnóstico IA (últimos 50)")
if os.path.exists("diagnostico_ia.csv"):
    df_diag = pd.read_csv("diagnostico_ia.csv")
    st.dataframe(df_diag.tail(50), use_container_width=True)
else:
    st.info("Nenhum diagnóstico encontrado ainda.")

