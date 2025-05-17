# bot_core_v53.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import os
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from ta.volatility import BollingerBands, AverageTrueRange
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    "API_TOKEN": "BZsX6ngxvhEBQOL",
    "SYMBOL": "stpRNG",
    "STAKE_INICIAL": 1,
    "STAKE_MAX": 100,
    "MARTINGALE_MULT": 2,
    "DURATION": 2,
    "LIMIAR_CONF": 0.7
}

CAMINHO_CSV = "dados_treinamento.csv"
DIAGNOSTICO_CSV = "diagnostico_ia.csv"
tick_history = deque(maxlen=100)

estatisticas = {
    "wins": 0,
    "losses": 0,
    "bloqueios": 0,
    "acertos": 0,
    "erros": 0,
    "lucro_total": 0.0
}

ia = None
ia_contexto = None
features_usadas = []
features_contexto = []

# Modelos

class ModeloMLPPyTorch(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        return self.rede(x)

class IA_PyTorch:
    def __init__(self, input_cols):
        self.input_cols = input_cols
        self.modelo = ModeloMLPPyTorch(len(input_cols))
        self.scaler = StandardScaler()
        self.optim = torch.optim.Adam(self.modelo.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def treinar(self, df):
        df = shuffle(df)
        X = df[self.input_cols].fillna(0).values
        y = df["resultado"].map({"won": 1, "lost": 0}).values
        streaks = df.get("streak_lost", pd.Series([0]*len(df))).values
        pesos = np.where(streaks >= 2, 2.0, 1.0)

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        pesos_tensor = torch.tensor(pesos, dtype=torch.float32)

        for _ in range(30):
            self.modelo.train()
            self.optim.zero_grad()
            out = self.modelo(X_tensor)
            losses = self.loss_fn(out, y_tensor)
            loss = (losses * pesos_tensor).mean()
            loss.backward()
            self.optim.step()

    def prever(self, linha):
        for col in self.input_cols:
            if col not in linha.columns:
                linha[col] = 0
        X = linha[self.input_cols].fillna(0).values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(self.modelo(X_tensor), dim=1).numpy()[0]
        return ('won' if probs[1] > probs[0] else 'lost', max(probs))

class EnsembleIA:
    def __init__(self, cols):
        self.modelos = [IA_PyTorch(cols) for _ in range(3)]
        self.cols = cols

    def treinar(self, df):
        for m in self.modelos:
            m.treinar(df)

    def prever(self, linha):
        resultados = [m.prever(linha) for m in self.modelos]
        classes = [r[0] for r in resultados]
        confs = [r[1] for r in resultados]
        return ("won" if all(c == "won" for c in classes) else "lost", np.mean(confs))

class IA_Contexto:
    def __init__(self, cols):
        self.input_cols = cols
        self.modelo = ModeloMLPPyTorch(len(cols))
        self.scaler = StandardScaler()
        self.optim = torch.optim.Adam(self.modelo.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def treinar(self, df):
        df = shuffle(df)
        X = df[self.input_cols].fillna(0).values
        y = df["contexto"].map({"seguro": 1, "arriscado": 0}).values
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        for _ in range(30):
            self.modelo.train()
            self.optim.zero_grad()
            out = self.modelo(X_tensor)
            loss = self.loss_fn(out, y_tensor)
            loss.mean().backward()
            self.optim.step()

    def prever(self, linha):
        for col in self.input_cols:
            if col not in linha.columns:
                linha[col] = 0
        X = linha[self.input_cols].fillna(0).values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(self.modelo(X_tensor), dim=1).numpy()[0]
        return ("seguro" if probs[1] >= probs[0] else "arriscado", max(probs))

def extrair_indicadores(ticks):
    df = pd.DataFrame(ticks, columns=["close"])
    df["high"] = df["close"] + 0.01
    df["low"] = df["close"] - 0.01
    bb = BollingerBands(df["close"])
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["var"] = df["close"].pct_change().rolling(window=5).std()
    df["ema_3"] = df["close"].ewm(span=3, adjust=False).mean()
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["delta_1"] = df["close"].diff(1)
    df["delta_2"] = df["close"].diff(2)
    df["ret_3"] = df["close"].pct_change(periods=3)
    equals_5 = sum([1 for i in range(1, 6) if df["close"].iloc[-i] == df["close"].iloc[-i-1]])
    df["equals_pct_5"] = equals_5 / 5.0
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=3)
    df["atr_3"] = atr.average_true_range()
    return df.iloc[-1].dropna()

def carregar_dados():
    return pd.read_csv(CAMINHO_CSV) if os.path.exists(CAMINHO_CSV) else pd.DataFrame()

def registrar_dados(ind, resultado):
    global ia, ia_contexto, features_usadas, features_contexto
    dados = carregar_dados()
    nova = ind.to_frame().T
    nova["resultado"] = resultado
    streak = sum(1 for r in reversed(dados["resultado"].tolist()) if r == "lost") if "resultado" in dados else 0
    nova["streak_lost"] = streak + 1 if resultado == "lost" else 0
    nova["contexto"] = "arriscado" if streak >= 2 else "seguro"
    dados = pd.concat([dados, nova], ignore_index=True)
    dados.to_csv(CAMINHO_CSV, index=False)
    if ia is None and len(dados) >= 100:
        features_usadas = list(dados.columns.drop(["resultado", "contexto"]))
        ia = EnsembleIA(features_usadas)
        ia.treinar(dados)
    if "contexto" in dados.columns:
        features_contexto = list(dados.columns.drop(["resultado", "contexto"]))
        ia_contexto = IA_Contexto(features_contexto)
        ia_contexto.treinar(dados)
    return dados

def iniciar_bot_em_thread():
    thread = threading.Thread(target=lambda: asyncio.run(executar_bot()), daemon=True)
    thread.start()

async def executar_bot():
    global ia, ia_contexto
    uri = "wss://ws.derivws.com/websockets/v3?app_id=75766"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"authorize": CONFIG["API_TOKEN"]}))
        await ws.send(json.dumps({"ticks": CONFIG["SYMBOL"], "subscribe": 1}))
        while True:
            msg = await ws.recv()
            tick = float(json.loads(msg)["tick"]["quote"])
            tick_history.append(tick)
            if len(tick_history) < 30:
                continue
            indicadores = extrair_indicadores(list(tick_history))
            if indicadores.isnull().any():
                continue
            direcao = "RISE" if tick > np.mean(list(tick_history)[-10:]) else "FALL"
            if not ia:
                registrar_dados(indicadores, "lost")
                continue
            predito, confianca = ia.prever(pd.DataFrame([indicadores]))
            seguro = True
            if ia_contexto:
                ctx, _ = ia_contexto.prever(pd.DataFrame([indicadores]))
                seguro = ctx == "seguro"
            permitir = predito == "won" and confianca >= CONFIG["LIMIAR_CONF"] and seguro
            registrar_dados(indicadores, "won" if permitir else "lost")
