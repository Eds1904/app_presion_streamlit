# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:45:25 2025

@author: XEXS23
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Funciones ---
def suavizar_picos(P, distance=5, prominence=100):
    peaks, _ = find_peaks(P, distance=distance, prominence=prominence)
    P_suav = P.copy()
    for idx in peaks:
        if 1 <= idx < len(P) - 1:
            P_suav[idx] = (P[idx - 1] + P[idx + 1]) / 2
        elif idx == 0:
            P_suav[idx] = P[1]
        elif idx == len(P) - 1:
            P_suav[idx] = P[-2]
    return P_suav, peaks

def estimacion_ito(P_suav, window=6):
    delta_P = np.diff(P_suav)
    mu = np.mean(delta_P)
    sigma = np.std(delta_P)
    mu_local = pd.Series(delta_P).rolling(window=window).mean()
    return mu, sigma, mu_local

def simular_mc(ultimo_valor, mu, sigma, horizonte=48, n_sim=10000, umbral=600, objetivo=0.638):
    tray = np.zeros((horizonte, n_sim))
    tray[0] = ultimo_valor
    for i in range(1, horizonte):
        ruido = np.random.normal(0, 1, n_sim)
        tray[i] = tray[i-1] + mu + sigma * np.sqrt(1) * ruido
        tray[i] = np.maximum(tray[i], 0)
    prob = (tray >= umbral).any(axis=0).mean()
    tiempo_obj = next((h+1 for h in range(horizonte)
                      if (tray[:h+1] >= umbral).any(axis=0).mean() >= objetivo), None)
    return tray, prob, tiempo_obj

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🔧 Análisis de Presión de Pozo")

archivo = st.file_uploader("📁 Subí tu archivo CSV", type=["csv"])

if archivo:
    try:
        df = pd.read_csv(archivo, sep=";", decimal=",", encoding="latin1")
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
        df["Presion"] = pd.to_numeric(df["Presion"], errors="coerce")
        df.dropna(subset=["TimeStamp", "Presion"], inplace=True)
        df.sort_values("TimeStamp", inplace=True)

        st.success("Archivo cargado correctamente ✅")
        st.dataframe(df.head())

        # --- Análisis ---
        P = df["Presion"].values
        t = df["TimeStamp"]
        P_suav, peaks = suavizar_picos(P)
        mu, sigma, mu_local = estimacion_ito(P_suav)

        st.subheader("📈 Presión y Deriva (Itô)")
        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(t, P, label="Presión original", color='steelblue')
        ax.plot(t[1:], mu_local, label="Deriva local", linestyle='--', color='orange')
        ax.plot(t[peaks], P[peaks], "x", label="Picos detectados", color='red')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # --- Monte Carlo ---
        st.subheader("🎲 Simulación Monte Carlo")
        umbral = st.slider("Umbral de presión (psi)", 0, 2000, 600, 50)
        horizonte = st.slider("Horizonte (horas)", 12, 168, 48, 12)
        n_sim = st.slider("Simulaciones", 1000, 20000, 5000, 1000)

        tray, prob, tiempo_obj = simular_mc(P_suav[-1], mu, sigma, horizonte, n_sim, umbral)

        st.markdown(f"**Deriva estimada (mu):** `{mu:.3f} psi/h`")
        st.markdown(f"**Volatilidad (sigma):** `{sigma:.3f} psi/sqrt(h)`")
        st.markdown(f"**Prob. de superar {umbral} psi:** `{prob:.2%}`")
        if tiempo_obj:
            st.markdown(f"**Tiempo estimado para superar {umbral} psi con 63.8% de probabilidad:** `{tiempo_obj} horas`")
        else:
            st.markdown(f"⚠️ No se alcanza esa probabilidad en las próximas {horizonte} horas.")

        # --- Gráfico de trayectorias
        fig2, ax2 = plt.subplots(figsize=(14,5))
        for i in range(min(200, n_sim)):
            ax2.plot(tray[:, i], color="gray", alpha=0.05)
        ax2.plot(np.median(tray, axis=1), color="blue", linewidth=2, label="Mediana")
        ax2.axhline(umbral, color="red", linestyle="--", label=f"Umbral = {umbral}")
        ax2.set_title("Trayectorias simuladas (Monte Carlo)")
        ax2.set_xlabel("Horas futuras")
        ax2.set_ylabel("Presión (psi)")
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
else:
    st.info("Esperando que subas un archivo...")

