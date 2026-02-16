import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import time
from datetime import datetime

# ==========================================================
# DATABASES
# ==========================================================

PROTEIN_DB = pd.DataFrame([
    {'Protein': 'BSA', 'MW_kDa': 66.5, 'pI': 4.7, 'Diff_Coeff': 6.0e-11, 'Sigma_Pa': 0.15},
    {'Protein': 'IgG', 'MW_kDa': 150.0, 'pI': 7.2, 'Diff_Coeff': 4.0e-11, 'Sigma_Pa': 0.10},
    {'Protein': 'β-gal', 'MW_kDa': 465.0, 'pI': 4.6, 'Diff_Coeff': 2.0e-11, 'Sigma_Pa': 0.08},
    {'Protein': 'Lysozyme', 'MW_kDa': 14.3, 'pI': 11.3, 'Diff_Coeff': 1.1e-10, 'Sigma_Pa': 0.25},
    {'Protein': 'Insulin', 'MW_kDa': 5.8, 'pI': 5.3, 'Diff_Coeff': 1.5e-10, 'Sigma_Pa': 0.30},
    {'Protein': 'EGFP', 'MW_kDa': 26.9, 'pI': 5.6, 'Diff_Coeff': 8.7e-11, 'Sigma_Pa': 0.18},
    {'Protein': 'mAb', 'MW_kDa': 145.0, 'pI': 8.5, 'Diff_Coeff': 4.2e-11, 'Sigma_Pa': 0.12}
])

MATERIAL_DB = pd.DataFrame({
    'Material': ['PDMS', 'PMMA', 'Hydrogel', 'Resin', 'SiO2'],
    'Surface_Energy': [19.8, 41.0, 65.0, 38.0, 72.0],
    'Thermal_Exp': [310, 70, 450, 120, 0.5],
    'Adsorption_Idx': [0.8, 0.6, 0.05, 0.7, 0.02],
    'Bio_Score': [0.85, 0.75, 0.95, 0.70, 0.90]
})

# ==========================================================
# ENGINE
# ==========================================================

class MicrofluidicMaster:

    def __init__(self, flow, w, h, l, temp, mat, protein):

        self.flow = max(0.1, flow)
        self.w = max(10, w)
        self.h = max(10, h)
        self.l = max(500, l)

        self.temp = temp
        self.mat = mat
        self.protein = protein

        self.p = PROTEIN_DB[PROTEIN_DB.Protein == protein].iloc[0]
        self.m = MATERIAL_DB[MATERIAL_DB.Material == mat].iloc[0]

        self.run_engine()


    def run_engine(self):

        w = self.w * 1e-6
        h = self.h * 1e-6
        Q = (self.flow * 1e-9) / 60

        rho = 1000
        mu = 0.001

        self.v = Q / (w*h)
        self.Dh = (2*w*h)/(w+h)

        self.Re = (rho*self.v*self.Dh)/mu
        self.tau = (6*mu*Q)/(w*h**2)

        q_lam = 1 if self.Re < 250 else max(0,1-(self.Re/2300))
        q_temp = max(0,1-abs(37-self.temp)/60)

        self.dqi = 0.6*q_lam + 0.4*q_temp

        q_shear = 1 if self.tau < self.p.Sigma_Pa else np.exp(-3*(self.tau-self.p.Sigma_Pa))
        res = (w*h*(self.l*1e-6))/Q
        q_res = min(1,res/5)

        self.psq = 0.3*self.dqi + 0.4*q_shear + 0.3*q_res

        q_bio = self.m.Bio_Score
        q_ads = 1 - self.m.Adsorption_Idx

        self.msq = 0.7*q_bio + 0.3*q_ads

        self.fitness = 0.35*self.dqi + 0.4*self.psq + 0.25*self.msq


# ==========================================================
# APP CONFIG
# ==========================================================

st.set_page_config("AI Microfluidic Master", layout="wide")

if "run" not in st.session_state:
    st.session_state.run = False

if "sim" not in st.session_state:
    st.session_state.sim = []

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🧪 System Control")

protein = st.sidebar.selectbox("Protein", PROTEIN_DB.Protein)

material = st.sidebar.selectbox("Material", MATERIAL_DB.Material)

flow = st.sidebar.slider("Flow (µL/min)",0.1,200.0,10.0,key="flow")

width = st.sidebar.slider("Width (µm)",10,1000,150,key="width")

height = st.sidebar.slider("Height (µm)",10,500,50,key="height")

length = st.sidebar.slider("Length (µm)",500,50000,5000,key="length")

temp = st.sidebar.slider("Temperature (°C)",4,80,37)


if st.sidebar.button("🚀 Run Simulation"):
    st.session_state.run = True


if not st.session_state.run:
    st.info("Configure parameters and click Run Simulation")
    st.stop()


# ==========================================================
# ENGINE INIT
# ==========================================================

eng = MicrofluidicMaster(flow,width,height,length,temp,material,protein)

# ==========================================================
# UI
# ==========================================================

st.title("🧬 AI Microfluidic Purification Master")

tabs = st.tabs([
    "📋 Database",
    "🧪 Design Lab",
    "📡 Live Run",
    "⚡ Stress",
    "📊 Analytics"
])

# ==========================================================
# TAB 1
# ==========================================================

with tabs[0]:

    st.subheader("Reference Databases")

    st.dataframe(PROTEIN_DB)
    st.dataframe(MATERIAL_DB)


# ==========================================================
# TAB 2
# ==========================================================

with tabs[1]:

    c1,c2 = st.columns([1,2])

    with c1:

        st.metric("Fitness",f"{eng.fitness:.3f}")
        st.progress(eng.fitness)

        st.metric("DQI",f"{eng.dqi:.2f}")
        st.metric("PSQ",f"{eng.psq:.2f}")
        st.metric("MSQ",f"{eng.msq:.2f}")


    with c2:

        st.write(f"Reynolds: {eng.Re:.1f}")
        st.write(f"Shear: {eng.tau:.4f} Pa")


        if eng.tau > eng.p.Sigma_Pa:
            st.error("Shear too high!")
        elif eng.Re > 2000:
            st.error("Turbulence!")
        elif eng.Re > 250:
            st.warning("Transition flow")
        else:
            st.success("Stable laminar flow")


        st.subheader("🧭 Optimization Advisor")

        if eng.Re > 250:
            st.write(f"🔻 Reduce flow to ~{flow*0.7:.1f}")

        if eng.tau > eng.p.Sigma_Pa:
            st.write(f"📐 Increase width to ~{int(width*1.4)}")
            st.write(f"📏 Increase height to ~{int(height*1.3)}")

        if eng.psq < 0.7:
            st.write(f"📏 Increase length to ~{int(length*1.2)}")



# ==========================================================
# TAB 3
# ==========================================================

with tabs[2]:

    st.subheader("🧪 Live Purification Run")

    a,b,c = st.columns(3)

    if a.button("▶️ Start"):
        st.session_state.live = True

    if b.button("⏸️ Stop"):
        st.session_state.live = False

    if c.button("🗑️ Reset"):
        st.session_state.sim = []


    plot = st.empty()


    if "live" not in st.session_state:
        st.session_state.live = False


    if st.session_state.live:

        for i in range(60):

            if not st.session_state.live:
                break

            s = len(st.session_state.sim)

            if s < 15:
                val = eng.psq*(s/15)

            elif eng.tau < eng.p.Sigma_Pa:
                val = eng.psq + np.random.normal(0,0.01)

            else:
                val = eng.psq*np.exp(-0.15*(s-15))


            val = max(0,min(1,val))


            st.session_state.sim.append({
                "Step":s,
                "Yield":val
            })


            df = pd.DataFrame(st.session_state.sim)

            fig = px.line(df,x="Step",y="Yield",
                          title="Recovery Curve",
                          template="plotly_dark")

            fig.update_layout(yaxis=dict(range=[0,1]))

            plot.plotly_chart(fig,use_container_width=True)

            time.sleep(0.1)


# ==========================================================
# TAB 4
# ==========================================================

with tabs[3]:

    if st.button("Run Stress Scan"):

        flows = np.linspace(0.1,200,80)

        out = []

        for f in flows:

            t = MicrofluidicMaster(f,width,height,length,temp,material,protein)

            out.append({
                "Flow":f,
                "Shear":t.tau,
                "Fitness":t.fitness
            })


        df = pd.DataFrame(out)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.Flow,y=df.Shear,name="Shear",line=dict(color="red")))
        fig.add_trace(go.Scatter(x=df.Flow,y=df.Fitness,name="Fitness",line=dict(color="cyan")))

        fig.add_hline(y=eng.p.Sigma_Pa,line_dash="dash",
                      annotation_text="Protein Limit")

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig,use_container_width=True)



# ==========================================================
# TAB 5
# ==========================================================

with tabs[4]:

    st.subheader("Sensitivity & Landscape")

    st.write("High Reynolds → Reduce Flow")
    st.write("High Shear → Increase Width/Height")
    st.write("Low Recovery → Increase Length")


    x = np.linspace(50,800,25)
    y = np.linspace(1,150,25)

    z = []

    for f in y:

        row = []

        for w in x:

            row.append(
                MicrofluidicMaster(
                    f,w,height,length,temp,material,protein
                ).fitness
            )

        z.append(row)


    fig = go.Figure(
        data=[go.Surface(z=z,x=x,y=y,colorscale="Viridis")]
    )


    fig.update_layout(
        title="Optimization Surface",
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Flow",
            zaxis_title="Fitness"
        )
    )


    st.plotly_chart(fig,use_container_width=True)


# ==========================================================

st.caption("v3.1 | AI Microfluidic Digital")
