# ==========================================================
# AI MICROFLUIDIC PURIFICATION SYSTEM
# Version 7.0 — Formal Research Prototype
# Author: Himani
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Microfluidic Purification Platform",
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
# MICROFLUIDIC ENGINE
# ==========================================================

class MicrofluidicEngine:

    def __init__(self, flow, width, height, length, temp, material, protein):

        self.flow = max(flow, 0.1)
        self.width = max(width, 10)
        self.height = max(height, 10)
        self.length = max(length, 500)

        self.temp = temp
        self.material = material
        self.protein = protein

        self.p = PROTEIN_DB.query("Protein == @protein").iloc[0]
        self.m = MATERIAL_DB.query("Material == @material").iloc[0]

        self.solve()


    def solve(self):

        w = self.width * 1e-6
        h = self.height * 1e-6
        Q = (self.flow * 1e-9) / 60

        rho = 1000
        mu = 0.001

        self.velocity = Q / (w * h)
        self.hydraulic_diameter = (2 * w * h) / (w + h)

        self.Re = rho * self.velocity * self.hydraulic_diameter / mu
        self.shear = (6 * mu * Q) / (w * h**2)

        laminar_quality = 1 if self.Re < 250 else max(0, 1 - self.Re / 2300)
        temperature_quality = max(0, 1 - abs(37 - self.temp) / 60)

        self.dqi = 0.6 * laminar_quality + 0.4 * temperature_quality

        shear_quality = (
            1 if self.shear < self.p.Sigma
            else np.exp(-3 * (self.shear - self.p.Sigma))
        )

        residence = (w * h * self.length * 1e-6) / Q
        residence_quality = min(1, residence / 5)

        self.psq = (
            0.3 * self.dqi +
            0.4 * shear_quality +
            0.3 * residence_quality
        )

        bio_quality = self.m.Bio
        adsorption_quality = 1 - self.m.Ads

        self.msq = 0.7 * bio_quality + 0.3 * adsorption_quality

        self.fitness = (
            0.35 * self.dqi +
            0.4 * self.psq +
            0.25 * self.msq
        )


# ==========================================================
# SENSOR SIMULATION (DATA-ONLY)
# ==========================================================

class CameraSensor:

    def __init__(self):
        self.last_clarity = np.random.uniform(0.6, 0.8)

    def read(self):

        zoom = np.random.uniform(3, 8)

        drift = np.random.normal(0, 0.03)

        clarity = self.last_clarity + drift
        clarity = np.clip(clarity, 0.4, 0.95)

        self.last_clarity = clarity

        return zoom, clarity


class LiDARSensor:

    def __init__(self):
        self.last_depth = None

    def read(self, height):

        if self.last_depth is None:
            self.last_depth = height

        drift = np.random.normal(0, 2)

        depth = self.last_depth + drift

        depth = np.clip(depth, height * 0.7, height * 1.3)

        self.last_depth = depth

        return depth


class ProteinVisionSensor:

    def __init__(self):
        self.last_visibility = np.random.uniform(0.6, 0.8)

    def read(self):

        drift = np.random.normal(0, 0.03)

        visibility = self.last_visibility + drift

        visibility = np.clip(visibility, 0.4, 0.95)

        self.last_visibility = visibility

        return visibility


class SeparationSensor:

    def __init__(self):
        self.last_purity = np.random.uniform(0.6, 0.75)

    def read(self):

        drift = np.random.normal(0, 0.04)

        purity = self.last_purity + drift

        purity = np.clip(purity, 0.3, 0.95)

        self.last_purity = purity

        return purity


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "run": False,
    "live": False,
    "sensor_log": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# SIDEBAR — SYSTEM CONTROL
# ==========================================================

st.sidebar.title("System Control Panel")

protein = st.sidebar.selectbox("Target Protein", PROTEIN_DB.Protein)

material = st.sidebar.selectbox("Channel Material", MATERIAL_DB.Material)

flow = st.sidebar.slider("Flow Rate (µL/min)", 0.1, 200.0, 10.0)

width = st.sidebar.slider("Channel Width (µm)", 10, 1000, 150)

height = st.sidebar.slider("Channel Height (µm)", 10, 500, 50)

length = st.sidebar.slider("Channel Length (µm)", 500, 50000, 5000)

temp = st.sidebar.slider("Operating Temperature (°C)", 4, 80, 37)


st.sidebar.markdown("---")

if st.sidebar.button("Initialize Simulation", use_container_width=True):
    st.session_state.run = True


# ==========================================================
# LANDING PAGE
# ==========================================================

if not st.session_state.run:

    st.title("Microfluidic Purification Digital Platform")

    st.info("""
    This system provides an integrated digital twin,
    sensor simulation, and analytical environment
    for microfluidic protein purification research.
    """)

    st.stop()


# ==========================================================
# INITIALIZATION
# ==========================================================

engine = MicrofluidicEngine(
    flow, width, height, length,
    temp, material, protein
)

camera = CameraSensor()
lidar = LiDARSensor()
vision = ProteinVisionSensor()
separation = SeparationSensor()


# ==========================================================
# HEADER
# ==========================================================

st.title("Microfluidic Purification System")
st.caption("Research and Development Prototype")


# ==========================================================
# TABS
# ==========================================================

tabs = st.tabs([
    "Reference Databases",
    "Design and Analysis",
    "Imaging and Sensors",
    "Stress Evaluation",
    "Optimization Analysis"
])


# ==========================================================
# TAB 1 — DATABASES
# ==========================================================

with tabs[0]:

    st.subheader("Protein Database")
    st.dataframe(PROTEIN_DB, use_container_width=True)

    st.subheader("Material Database")
    st.dataframe(MATERIAL_DB, use_container_width=True)


# ==========================================================
# TAB 2 — DESIGN
# ==========================================================

with tabs[1]:

    c1, c2 = st.columns([1,2])


    with c1:

        st.subheader("System Performance")

        st.metric("Fitness Index", f"{engine.fitness:.3f}")
        st.progress(engine.fitness)

        st.metric("DQI", f"{engine.dqi:.2f}")
        st.metric("PSQ", f"{engine.psq:.2f}")
        st.metric("MSQ", f"{engine.msq:.2f}")


    with c2:

        st.subheader("Hydrodynamic Status")

        st.write(f"Reynolds Number: {engine.Re:.1f}")
        st.write(f"Shear Stress: {engine.shear:.4f} Pa")


        if engine.shear > engine.p.Sigma:
            st.error("Shear stress exceeds protein tolerance")

        elif engine.Re > 2000:
            st.error("Turbulent flow regime")

        elif engine.Re > 250:
            st.warning("Transitional flow regime")

        else:
            st.success("Stable laminar flow")


        st.subheader("Design Recommendations")

        if engine.Re > 250:
            st.write("Reduce flow rate")

        if engine.shear > engine.p.Sigma:
            st.write("Increase channel dimensions")

        if engine.psq < 0.7:
            st.write("Increase channel length")


# ==========================================================
# TAB 3 — SENSORS
# ==========================================================

with tabs[2]:

    st.subheader("Imaging and Sensor Monitoring")

    a, b, c = st.columns(3)

    if a.button("Start Acquisition"):
        st.session_state.live = True

    if b.button("Stop Acquisition"):
        st.session_state.live = False

    if c.button("Clear Log"):
        st.session_state.sensor_log = []


    cam_box = st.empty()
    lidar_box = st.empty()
    ai_box = st.empty()


    if st.session_state.live:

        for _ in range(200):

            if not st.session_state.live:
                break


            zoom, clarity = camera.read()

            depth = lidar.read(height)

            visibility = vision.read()

            purity = separation.read()


            record = {
                "Zoom": zoom,
                "Clarity": clarity,
                "Depth": depth,
                "Visibility": visibility,
                "Purity": purity
            }

            st.session_state.sensor_log.append(record)


            cam_box.metric("Camera Zoom (x)", f"{zoom:.2f}")
            cam_box.progress(clarity)

            lidar_box.metric("LiDAR Depth (µm)", f"{depth:.1f}")

            ai_box.metric("Protein Visibility", f"{visibility:.2f}")
            ai_box.metric("Separation Purity", f"{purity:.2f}")


            if purity > 0.85:
                st.success("High purity separation")

            elif purity > 0.65:
                st.warning("Moderate separation")

            else:
                st.error("Low separation efficiency")


            time.sleep(0.25)


# ==========================================================
# TAB 4 — STRESS TEST
# ==========================================================

with tabs[3]:

    st.subheader("Flow Stress Evaluation")

    if st.button("Execute Stress Test"):

        with st.spinner("Processing parameter sweep..."):

            flows = np.linspace(0.5, 200, 80)

            results = []


            for f in flows:

                t = MicrofluidicEngine(
                    f, width, height, length,
                    temp, material, protein
                )

                results.append([
                    f, t.shear, t.Re, t.fitness
                ])


            df = pd.DataFrame(
                results,
                columns=["Flow","Shear","Reynolds","Fitness"]
            )


        st.success("Stress evaluation completed")

        st.dataframe(df, use_container_width=True)


        unsafe = df[
            (df.Shear > engine.p.Sigma) |
            (df.Reynolds > 2000)
        ]


        st.subheader("Unstable Operating Conditions")

        if len(unsafe) > 0:
            st.dataframe(unsafe, use_container_width=True)

        else:
            st.success("All evaluated conditions are stable")


# ==========================================================
# TAB 5 — ANALYTICS
# ==========================================================

with tabs[4]:

    st.subheader("Design Optimization Landscape")


    st.info("""
    Flow instability and excessive shear
    reduce separation efficiency.

    This surface represents the combined
    performance index across parameters.
    """)


    x = np.linspace(50, 800, 25)
    y = np.linspace(1, 150, 25)

    z = []


    for f in y:

        row = []

        for w in x:

            val = MicrofluidicEngine(
                f, w, height, length,
                temp, material, protein
            ).fitness

            row.append(val)

        z.append(row)


    fig = go.Figure(
        data=[go.Surface(z=z, x=x, y=y)]
    )


    fig.update_layout(
        scene=dict(
            xaxis_title="Width (µm)",
            yaxis_title="Flow Rate (µL/min)",
            zaxis_title="Fitness Index"
        )
    )


    st.plotly_chart(fig, use_container_width=True)


# ==========================================================

st.caption("Version 7.0 — Formal Microfluidic Research Platform")
