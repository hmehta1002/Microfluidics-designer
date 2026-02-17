# ==========================================================
# AI MICROFLUIDIC PURIFICATION SYSTEM v6.0
# Complete Digital Twin + Sensor Platform
# Author: Himani
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Microfluidic Master",
    page_icon="🧬",
    layout="wide"
)

# ==========================================================
# DATABASES
# ==========================================================

PROTEIN_DB = pd.DataFrame([
    {"Protein":"BSA","MW":66.5,"pI":4.7,"Diff":6e-11,"Sigma":0.15},
    {"Protein":"IgG","MW":150,"pI":7.2,"Diff":4e-11,"Sigma":0.10},
    {"Protein":"β-gal","MW":465,"pI":4.6,"Diff":2e-11,"Sigma":0.08},
    {"Protein":"Lysozyme","MW":14.3,"pI":11.3,"Diff":1.1e-10,"Sigma":0.25},
    {"Protein":"Insulin","MW":5.8,"pI":5.3,"Diff":1.5e-10,"Sigma":0.30},
    {"Protein":"EGFP","MW":26.9,"pI":5.6,"Diff":8.7e-11,"Sigma":0.18},
    {"Protein":"mAb","MW":145,"pI":8.5,"Diff":4.2e-11,"Sigma":0.12}
])

MATERIAL_DB = pd.DataFrame([
    {"Material":"PDMS","Energy":19.8,"Ads":0.8,"Bio":0.85},
    {"Material":"PMMA","Energy":41,"Ads":0.6,"Bio":0.75},
    {"Material":"Hydrogel","Energy":65,"Ads":0.05,"Bio":0.95},
    {"Material":"Resin","Energy":38,"Ads":0.7,"Bio":0.70},
    {"Material":"SiO2","Energy":72,"Ads":0.02,"Bio":0.90}
])

# ==========================================================
# PHYSICS ENGINE
# ==========================================================

class MicrofluidicEngine:

    def __init__(self, flow,w,h,l,temp,mat,protein):

        self.flow = max(flow,0.1)
        self.w = max(w,10)
        self.h = max(h,10)
        self.l = max(l,500)

        self.temp = temp
        self.mat = mat
        self.protein = protein

        self.p = PROTEIN_DB.query("Protein==@protein").iloc[0]
        self.m = MATERIAL_DB.query("Material==@mat").iloc[0]

        self.solve()


    def solve(self):

        w = self.w*1e-6
        h = self.h*1e-6
        Q = (self.flow*1e-9)/60

        rho = 1000
        mu = 0.001

        self.v = Q/(w*h)
        self.Dh = (2*w*h)/(w+h)

        self.Re = rho*self.v*self.Dh/mu
        self.tau = (6*mu*Q)/(w*h**2)

        lam = 1 if self.Re<250 else max(0,1-self.Re/2300)
        tq = max(0,1-abs(37-self.temp)/60)

        self.dqi = 0.6*lam+0.4*tq

        shear = 1 if self.tau<self.p.Sigma else np.exp(-3*(self.tau-self.p.Sigma))
        res = (w*h*self.l*1e-6)/Q
        resq = min(1,res/5)

        self.psq = 0.3*self.dqi+0.4*shear+0.3*resq

        bio = self.m.Bio
        ads = 1-self.m.Ads

        self.msq = 0.7*bio+0.3*ads

        self.fitness = (
            0.35*self.dqi +
            0.4*self.psq +
            0.25*self.msq
        )


# ==========================================================
# HARDWARE PLACEHOLDER LAYER
# ==========================================================

class CameraSensor:

    def read(self, psq):

        zoom = np.random.uniform(2,10)
        clarity = np.clip(psq + np.random.normal(0,0.05),0,1)

        return zoom,clarity


class LiDARSensor:

    def read(self, height):

        depth = np.random.uniform(
            height*0.7,
            height*1.3
        )

        noise = np.random.normal(0,1)

        return depth+noise


class ProteinVisionAI:

    def analyze(self, clarity, msq):

        visibility = np.clip(
            clarity*msq + np.random.normal(0,0.05),
            0,1
        )

        return visibility


class SeparationAI:

    def evaluate(self, visibility, fitness):

        purity = np.clip(
            visibility*fitness + np.random.normal(0,0.03),
            0,1
        )

        return purity


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "run":False,
    "live":False,
    "sim":[],
    "sensor_log":[]
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🧪 Control Panel")

protein = st.sidebar.selectbox("Protein",PROTEIN_DB.Protein)
material = st.sidebar.selectbox("Material",MATERIAL_DB.Material)

flow = st.sidebar.slider("Flow (µL/min)",0.1,200.0,10.0)
width = st.sidebar.slider("Width (µm)",10,1000,150)
height = st.sidebar.slider("Height (µm)",10,500,50)
length = st.sidebar.slider("Length (µm)",500,50000,5000)
temp = st.sidebar.slider("Temperature (°C)",4,80,37)

st.sidebar.markdown("---")

if st.sidebar.button("🚀 Run Simulation",use_container_width=True):
    st.session_state.run=True


# ==========================================================
# START PAGE
# ==========================================================

if not st.session_state.run:

    st.title("🧬 AI Microfluidic Purification Platform")

    st.info("""
    Digital Twin + Imaging + AI + Optimization
    
    Configure → Simulate → Sense → Analyze → Optimize
    """)

    st.stop()


# ==========================================================
# ENGINE INIT
# ==========================================================

engine = MicrofluidicEngine(
    flow,width,height,length,temp,material,protein
)

camera = CameraSensor()
lidar = LiDARSensor()
vision = ProteinVisionAI()
sepAI = SeparationAI()


# ==========================================================
# HEADER
# ==========================================================

st.title("🧬 AI Microfluidic Purification Master")
st.caption("Digital Twin | Sensors | AI Control | Hardware Prototype")


# ==========================================================
# TABS
# ==========================================================

tabs = st.tabs([
    "📋 Databases",
    "🧪 Design Lab",
    "📡 Sensors",
    "⚡ Stress Test",
    "📊 Analytics"
])


# ==========================================================
# TAB 1 — DATABASE
# ==========================================================

with tabs[0]:

    st.subheader("Protein Library")
    st.dataframe(PROTEIN_DB,use_container_width=True)

    st.subheader("Material Library")
    st.dataframe(MATERIAL_DB,use_container_width=True)


# ==========================================================
# TAB 2 — DESIGN LAB
# ==========================================================

with tabs[1]:

    c1,c2 = st.columns([1,2])

    with c1:

        st.metric("Fitness",f"{engine.fitness:.3f}")
        st.progress(engine.fitness)

        st.metric("DQI",f"{engine.dqi:.2f}")
        st.metric("PSQ",f"{engine.psq:.2f}")
        st.metric("MSQ",f"{engine.msq:.2f}")


    with c2:

        st.subheader("Flow Physics")

        st.write(f"Reynolds: {engine.Re:.1f}")
        st.write(f"Shear: {engine.tau:.4f} Pa")

        if engine.tau>engine.p.Sigma:
            st.error("Protein Damage Risk")

        elif engine.Re>2000:
            st.error("Turbulence")

        elif engine.Re>250:
            st.warning("Transition Flow")

        else:
            st.success("Laminar Stable")


        st.subheader("Optimization Advisor")

        if engine.Re>250:
            st.write(f"Reduce Flow → {flow*0.7:.1f}")

        if engine.tau>engine.p.Sigma:
            st.write(f"Increase Width → {int(width*1.4)}")
            st.write(f"Increase Height → {int(height*1.3)}")

        if engine.psq<0.7:
            st.write(f"Increase Length → {int(length*1.2)}")


# ==========================================================
# TAB 3 — SENSOR DASHBOARD
# ==========================================================

with tabs[2]:

    st.subheader("📡 Imaging + LiDAR + AI System")

    a,b,c = st.columns(3)

    if a.button("▶ Start Scan"):
        st.session_state.live=True

    if b.button("⏸ Stop"):
        st.session_state.live=False

    if c.button("♻ Reset"):
        st.session_state.sensor_log=[]


    cam_box = st.empty()
    lidar_box = st.empty()
    ai_box = st.empty()


    if st.session_state.live:

        for i in range(150):

            if not st.session_state.live:
                break


            zoom,clarity = camera.read(engine.psq)

            depth = lidar.read(height)

            visibility = vision.analyze(
                clarity,engine.msq
            )

            purity = sepAI.evaluate(
                visibility,engine.fitness
            )


            record = {
                "Zoom":zoom,
                "Clarity":clarity,
                "Depth":depth,
                "Visibility":visibility,
                "Purity":purity
            }

            st.session_state.sensor_log.append(record)


            cam_box.metric("📷 Zoom",f"{zoom:.2f}x")
            cam_box.progress(clarity)

            lidar_box.metric("📡 Depth (µm)",f"{depth:.1f}")

            ai_box.metric("👁 Visibility",f"{visibility:.2f}")
            ai_box.metric("🧪 Purity",f"{purity:.2f}")


            if purity>0.85:
                st.success("High Purity Separation")

            elif purity>0.65:
                st.warning("Moderate Separation")

            else:
                st.error("Poor Separation")


            time.sleep(0.25)


# ==========================================================
# TAB 4 — STRESS TEST
# ==========================================================

with tabs[3]:

    st.subheader("⚡ Flow Stress Scanner")

    if st.button("Run Stress Scan"):

        with st.spinner("Running stress simulation..."):

            flows = np.linspace(0.5,200,80)

            rows=[]

            for f in flows:

                t = MicrofluidicEngine(
                    f,width,height,length,temp,material,protein
                )

                rows.append([
                    f,t.tau,t.Re,t.fitness
                ])


            df = pd.DataFrame(
                rows,
                columns=["Flow","Shear","Re","Fitness"]
            )


        st.success("Scan Complete")

        st.dataframe(df,use_container_width=True)

        danger = df[
            (df.Shear>engine.p.Sigma)|
            (df.Re>2000)
        ]


        st.subheader("Unsafe Operating Zone")

        if len(danger)>0:
            st.dataframe(danger,use_container_width=True)
        else:
            st.success("All Conditions Stable")


# ==========================================================
# TAB 5 — ANALYTICS
# ==========================================================

with tabs[4]:

    st.subheader("Optimization Landscape")

    st.info("""
    High Re → Reduce Flow  
    High Shear → Increase Channel Size  
    Low Purity → Increase Residence Time
    """)


    x = np.linspace(50,800,25)
    y = np.linspace(1,150,25)

    z=[]


    for f in y:

        row=[]

        for w in x:

            row.append(
                MicrofluidicEngine(
                    f,w,height,length,temp,material,protein
                ).fitness
            )

        z.append(row)


    fig = go.Figure(
        data=[go.Surface(z=z,x=x,y=y)]
    )


    fig.update_layout(
        scene=dict(
            xaxis_title="Width (µm)",
            yaxis_title="Flow (µL/min)",
            zaxis_title="Fitness"
        )
    )


    st.plotly_chart(fig,use_container_width=True)


# ==========================================================

st.caption("v6.0 | AI Microfluidic Digital Twin + Sensor Platform")
