# app.py

import streamlit as st
import pandas as pd
import os
from bot_core_v53 import CONFIG, estatisticas, iniciar_bot_em_thread

st.set_page_config(page_title="WebBot IA v53", layout="wide")
st.title("🧠 WebBot IA v53 - PyTorch + Ensemble")

st.sidebar.header("⚙️ Configurações do Bot")
CONFIG["STAKE_INICIAL"] = st.sidebar.number_input("Stake Inicial", 0.1, 100.0, 1.0)
CONFIG["STAKE_MAX"] = st.sidebar.number_input("Stake Máxima", 1.0, 1000.0, 100.0)
CONFIG["MARTINGALE_MULT"] = st.sidebar.slider("Multiplicador Martingale", 1.0, 10.0, 2.0)
CONFIG["DURATION"] = st.sidebar.slider("Ticks", 1, 10, 2)
CONFIG["LIMIAR_CONF"] = st.sidebar.slider("Confiança IA", 0.5, 1.0, 0.7)

if st.sidebar.button("🚀 Iniciar Bot IA"):
    iniciar_bot_em_thread()
    st.sidebar.success("Bot iniciado.")

st.sidebar.markdown("---")
st.sidebar.subheader("📁 CSVs")
if os.path.exists("diagnostico_ia.csv"):
    with open("diagnostico_ia.csv", "rb") as f:
        st.sidebar.download_button("Download Diagnóstico", f, "diagnostico_ia.csv")

if os.path.exists("dados_treinamento.csv"):
    with open("dados_treinamento.csv", "rb") as f:
        st.sidebar.download_button("Download Treino", f, "dados_treinamento.csv")

st.subheader("📊 Estatísticas")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acertos", estatisticas["acertos"])
col2.metric("Erros", estatisticas["erros"])
col3.metric("Bloqueios", estatisticas["bloqueios"])
col4.metric("Lucro", f"${estatisticas['lucro_total']:.2f}")

st.subheader("📋 Diagnóstico")
try:
    df = pd.read_csv("diagnostico_ia.csv", encoding="latin1")
    st.dataframe(df.tail(50), use_container_width=True)
except Exception as e:
    st.warning(f"Erro ao carregar diagnóstico: {e}")
