# ==========================================================
# AI MICROFLUIDIC PURIFICATION MASTER v4.0
# Author: Himani
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Microfluidic Master",
    layout="wide",
    page_icon="🧬"
)

# ==========================================================
# DATABASES
# ==========================================================

PROTEIN_DB = pd.DataFrame([
    {"Protein": "BSA", "MW": 66.5, "pI": 4.7, "Diff": 6e-11, "Sigma": 0.15},
    {"Protein": "IgG", "MW": 150, "pI": 7.2, "Diff": 4e-11, "Sigma": 0.10},
    {"Protein": "β-gal", "MW": 465, "pI": 4.6, "Diff": 2e-11, "Sigma": 0.08},
    {"Protein": "Lysozyme", "MW": 14.3, "pI": 11.3, "Diff": 1.1e-10, "Sigma": 0.25},
    {"Protein": "Insulin", "MW": 5.8, "pI": 5.3, "Diff": 1.5e-10, "Sigma": 0.30},
    {"Protein": "EGFP", "MW": 26.9, "pI": 5.6, "Diff": 8.7e-11, "Sigma": 0.18},
    {"Protein": "mAb", "MW": 145, "pI": 8.5, "Diff": 4.2e-11, "Sigma": 0.12},
])


MATERIAL_DB = pd.DataFrame([
    {"Material": "PDMS", "Energy": 19.8, "Ads": 0.8, "Bio": 0.85},
    {"Material": "PMMA", "Energy": 41, "Ads": 0.6, "Bio": 0.75},
    {"Material": "Hydrogel", "Energy": 65, "Ads": 0.05, "Bio": 0.95},
    {"Material": "Resin", "Energy": 38, "Ads": 0.7, "Bio": 0.70},
    {"Material": "SiO2", "Energy": 72, "Ads": 0.02, "Bio": 0.90},
])


# ==========================================================
# ENGINE
# ==========================================================

class MicrofluidicEngine:

    def __init__(self, flow, w, h, l, temp, mat, protein):

        self.flow = max(flow, 0.1)
        self.w = max(w, 10)
        self.h = max(h, 10)
        self.l = max(l, 500)

        self.temp = temp
        self.mat = mat
        self.protein = protein

        self.p = PROTEIN_DB.query("Protein == @protein").iloc[0]
        self.m = MATERIAL_DB.query("Material == @mat").iloc[0]

        self.compute()


    def compute(self):

        w = self.w * 1e-6
        h = self.h * 1e-6
        Q = (self.flow * 1e-9) / 60

        rho = 1000
        mu = 0.001

        self.v = Q / (w * h)
        self.Dh = (2 * w * h) / (w + h)

        self.Re = rho * self.v * self.Dh / mu
        self.tau = (6 * mu * Q) / (w * h**2)

        # DQI
        lam = 1 if self.Re < 250 else max(0, 1 - self.Re / 2300)
        temp_q = max(0, 1 - abs(37 - self.temp) / 60)

        self.dqi = 0.6 * lam + 0.4 * temp_q

        # PSQ
        shear_q = 1 if self.tau < self.p.Sigma else np.exp(-3*(self.tau-self.p.Sigma))
        res = (w * h * self.l * 1e-6) / Q
        res_q = min(1, res / 5)

        self.psq = 0.3*self.dqi + 0.4*shear_q + 0.3*res_q

        # MSQ
        bio = self.m.Bio
        ads = 1 - self.m.Ads

        self.msq = 0.7*bio + 0.3*ads

        # Fitness
        self.fitness = 0.35*self.dqi + 0.4*self.psq + 0.25*self.msq


# ==========================================================
# SESSION STATE
# ==========================================================

if "run" not in st.session_state:
    st.session_state.run = False

if "live" not in st.session_state:
    st.session_state.live = False

if "sim" not in st.session_state:
    st.session_state.sim = []


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🧪 Control Panel")

protein = st.sidebar.selectbox("Protein", PROTEIN_DB.Protein)
material = st.sidebar.selectbox("Material", MATERIAL_DB.Material)

flow = st.sidebar.slider("Flow (µL/min)", 0.1, 200.0, 10.0)
width = st.sidebar.slider("Width (µm)", 10, 1000, 150)
height = st.sidebar.slider("Height (µm)", 10, 500, 50)
length = st.sidebar.slider("Length (µm)", 500, 50000, 5000)
temp = st.sidebar.slider("Temperature (°C)", 4, 80, 37)


st.sidebar.markdown("---")

if st.sidebar.button("🚀 Run Simulation", use_container_width=True):
    st.session_state.run = True


# ==========================================================
# START SCREEN
# ==========================================================

if not st.session_state.run:

    st.title("🧬 AI Microfluidic Purification Platform")

    st.info("""
    Configure parameters → Run Simulation → Optimize Design
    
    This platform simulates protein purification efficiency
    using fluid dynamics + biomaterial models.
    """)

    st.stop()


# ==========================================================
# ENGINE INIT
# ==========================================================

engine = MicrofluidicEngine(
    flow, width, height, length, temp, material, protein
)


# ==========================================================
# HEADER
# ==========================================================

st.title("🧬 AI Microfluidic Purification Master")
st.caption("Digital Twin | Optimization Engine | Research Tool")


# ==========================================================
# TABS
# ==========================================================

tabs = st.tabs([
    "📋 Databases",
    "🧪 Design Lab",
    "📡 Live Run",
    "⚡ Stress Test",
    "📊 Analytics"
])


# ==========================================================
# TAB 1 — DATABASE
# ==========================================================

with tabs[0]:

    st.subheader("Protein Library")
    st.dataframe(PROTEIN_DB, use_container_width=True)

    st.subheader("Material Library")
    st.dataframe(MATERIAL_DB, use_container_width=True)


# ==========================================================
# TAB 2 — DESIGN LAB
# ==========================================================

with tabs[1]:

    c1, c2 = st.columns([1, 2])


    with c1:

        st.subheader("Performance")

        st.metric("Fitness", f"{engine.fitness:.3f}")
        st.progress(engine.fitness)

        st.metric("DQI", f"{engine.dqi:.2f}")
        st.metric("PSQ", f"{engine.psq:.2f}")
        st.metric("MSQ", f"{engine.msq:.2f}")


    with c2:

        st.subheader("Flow Physics")

        st.write(f"Reynolds: {engine.Re:.1f}")
        st.write(f"Shear: {engine.tau:.4f} Pa")


        if engine.tau > engine.p.Sigma:
            st.error("⚠️ Protein Damage Risk")

        elif engine.Re > 2000:
            st.error("⚠️ Turbulent Flow")

        elif engine.Re > 250:
            st.warning("⚠️ Transitional Flow")

        else:
            st.success("✅ Stable Laminar Flow")


        st.subheader("Optimization Advisor")

        if engine.Re > 250:
            st.write(f"🔻 Reduce Flow → {flow*0.7:.1f}")

        if engine.tau > engine.p.Sigma:
            st.write(f"📐 Increase Width → {int(width*1.4)} µm")
            st.write(f"📏 Increase Height → {int(height*1.3)} µm")

        if engine.psq < 0.7:
            st.write(f"📏 Increase Length → {int(length*1.2)} µm")


# ==========================================================
# TAB 3 — LIVE RUN
# ==========================================================

with tabs[2]:

    st.subheader("Live Purification Monitor")

    a, b, c = st.columns(3)

    if a.button("▶ Start"):
        st.session_state.live = True

    if b.button("⏸ Stop"):
        st.session_state.live = False

    if c.button("♻ Reset"):
        st.session_state.sim = []


    chart = st.empty()


    if st.session_state.live:

        for i in range(60):

            if not st.session_state.live:
                break

            s = len(st.session_state.sim)

            if s < 15:
                val = engine.psq * (s / 15)

            elif engine.tau < engine.p.Sigma:
                val = engine.psq + np.random.normal(0, 0.01)

            else:
                val = engine.psq * np.exp(-0.15*(s-15))


            val = np.clip(val, 0, 1)

            st.session_state.sim.append({
                "Step": s,
                "Yield": val
            })


            df = pd.DataFrame(st.session_state.sim)

            fig = px.line(
                df, x="Step", y="Yield",
                title="Recovery Curve"
            )

            fig.update_layout(yaxis=dict(range=[0,1]))

            chart.plotly_chart(fig, use_container_width=True)

            time.sleep(0.1)


# ==========================================================
# TAB 4 — STRESS TEST
# ==========================================================

with tabs[3]:

    if st.button("Run Stress Scan"):

        flows = np.linspace(0.1, 200, 80)

        data = []

        for f in flows:

            t = MicrofluidicEngine(
                f, width, height, length, temp, material, protein
            )

            data.append({
                "Flow": f,
                "Shear": t.tau,
                "Fitness": t.fitness
            })


        df = pd.DataFrame(data)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.Flow, y=df.Shear,
            name="Shear"
        ))

        fig.add_trace(go.Scatter(
            x=df.Flow, y=df.Fitness,
            name="Fitness"
        ))

        fig.add_hline(
            y=engine.p.Sigma,
            line_dash="dash",
            annotation_text="Protein Limit"
        )

        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# TAB 5 — ANALYTICS
# ==========================================================

with tabs[4]:

    st.subheader("Sensitivity Landscape")

    st.info("""
    High Reynolds → Reduce Flow  
    High Shear → Increase Dimensions  
    Low Yield → Increase Residence Time
    """)


    x = np.linspace(50, 800, 25)
    y = np.linspace(1, 150, 25)

    z = []


    for f in y:

        row = []

        for w in x:

            row.append(
                MicrofluidicEngine(
                    f, w, height, length, temp, material, protein
                ).fitness
            )

        z.append(row)


    fig = go.Figure(
        data=[go.Surface(z=z, x=x, y=y)]
    )


    fig.update_layout(
        title="Optimization Surface",
        scene=dict(
            xaxis_title="Width (µm)",
            yaxis_title="Flow (µL/min)",
            zaxis_title="Fitness"
        )
    )


    st.plotly_chart(fig, use_container_width=True)


# ==========================================================

st.caption("v4.0 | AI Microfluidic Digital Twin Platform")
