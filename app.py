# ==========================================================
# COMPLETE AI MICROFLUIDIC OPTIMIZATION PLATFORM
# Extended Version (Merged + Automated + Monitor)
# ==========================================================

import streamlit as st
import pandas as pd
import random
import math
import time


# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="AI Microfluidic Optimizer",
    layout="wide"
)

st.title("🧬 AI Microfluidic Optimization Platform")

st.markdown("""
Research-grade decision system for
microfluidic protein purification.

Includes:
• DQI / PSQ / MSQ Analysis
• Batch Experiments
• Genetic Optimization
• Virtual CFD
• Automated Optimization
• Live Monitoring (Simulated)
""")


# ==========================================================
# AI WEIGHTS
# ==========================================================

W_DQI = 0.35
W_PSQ = 0.40
W_MSQ = 0.25
BIAS = 0.05


# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("⚙️ Design Parameters")

protein = st.sidebar.text_input("Target Protein", "Albumin")

material = st.sidebar.selectbox(
    "Material",
    ["PDMS", "PMMA", "Hydrogel", "Resin"]
)

flow = st.sidebar.slider("Flow (µL/min)", 1.0, 50.0, 10.0)
width = st.sidebar.slider("Width (µm)", 50, 300, 120)
height = st.sidebar.slider("Height (µm)", 30, 200, 60)
length = st.sidebar.slider("Length (µm)", 500, 3000, 1500)
temp = st.sidebar.slider("Temperature (°C)", 20, 60, 37)

run_single = st.sidebar.button("▶ Single Run")
run_batch = st.sidebar.button("📊 Batch (25 Runs)")
run_ga = st.sidebar.button("🧬 Genetic Optimization")
run_cfd = st.sidebar.button("🌀 Virtual CFD")
run_auto = st.sidebar.button("🤖 Auto Optimize")
run_monitor = st.sidebar.button("📡 Live Monitor (Simulated)")


# ==========================================================
# CORE MODELS
# ==========================================================

def compute_dqi(flow, w, h, temp, material):

    w *= 1e-6
    h *= 1e-6

    Q = flow * 1e-9 / 60
    rho = 1000
    mu = 0.001

    A = w * h
    v = Q / A

    Dh = 2 * w * h / (w + h)

    Re = (rho * v * Dh) / mu
    tau = (6 * mu * Q) / (w * h**2)

    laminar = min(1, 200 / Re)
    shear_damage = math.exp(-0.02 * tau)
    temp_factor = max(0.6, 1 - abs(37 - temp) / 40)

    mat_factor = {
        "PDMS": 0.85,
        "PMMA": 0.75,
        "Hydrogel": 0.9,
        "Resin": 0.7
    }[material]

    dqi = (
        0.3 * laminar +
        0.25 * shear_damage +
        0.25 * temp_factor +
        0.2 * mat_factor
    )

    return round(dqi, 3), Re, tau


def compute_psq(dqi, flow, material, w, h, l):

    porosity = (w * h) / (500 * 500)

    affinity = {
        "PDMS": 0.7,
        "PMMA": 0.65,
        "Hydrogel": 0.9,
        "Resin": 0.6
    }[material]

    residence = max(0.5, 1 - flow / 70)
    length_factor = min(1, l / 1800)

    psq = (
        0.35 * dqi +
        0.30 * affinity +
        0.2 * residence +
        0.1 * length_factor +
        0.05 * porosity
    )

    return round(psq, 3), affinity, residence, length_factor


def compute_msq(material, temp, flow):

    props = {
        "PDMS": [0.9, 0.6, 0.85, 0.9, 0.7],
        "PMMA": [0.75, 0.7, 0.8, 0.85, 0.75],
        "Hydrogel": [0.95, 0.85, 0.65, 0.6, 0.9],
        "Resin": [0.7, 0.6, 0.9, 0.85, 0.6]
    }

    bio, ads, stab, fab, wet = props[material]

    temp_f = max(0.7, 1 - abs(37 - temp) / 50)
    flow_f = max(0.7, 1 - flow / 100)

    affinity = 1 - ads

    msq = (
        0.25 * bio +
        0.20 * affinity +
        0.20 * stab * temp_f +
        0.20 * fab +
        0.15 * wet * flow_f
    )

    return round(msq, 3)


def compute_fitness(dqi, psq, msq):

    return round(
        W_DQI * dqi +
        W_PSQ * psq +
        W_MSQ * msq +
        BIAS,
        3
    )


# ==========================================================
# ANALYSIS + OPTIMIZATION
# ==========================================================

def optimization_engine(dqi, psq, msq, Re, aff, res, lenf, material):

    plan = []

    if Re > 200:
        plan.append("Reduce flow")

    if aff < 0.7:
        plan.append("Switch to Hydrogel")

    if res < 0.7:
        plan.append("Reduce flow")

    if lenf < 0.7:
        plan.append("Increase length")

    if msq < 0.7:
        plan.append("Change material")

    if psq > 0.8:
        plan.append("Near optimal")

    return plan


def apply_optimization(flow, length, material, plan):

    f = flow
    l = length
    m = material

    for p in plan:

        if "Reduce flow" in p:
            f *= 0.8

        if "Increase length" in p:
            l *= 1.2

        if "Switch to Hydrogel" in p:
            m = "Hydrogel"

        if "Change material" in p:
            m = "PMMA"

    return f, l, m


# ==========================================================
# SENSOR SIMULATION
# ==========================================================

def simulate_sensors(flow, temp):

    f = flow * random.uniform(0.9, 1.1)
    t = temp + random.uniform(-0.7, 0.7)
    p = random.uniform(30, 60)

    return round(f,2), round(t,2), round(p,2)


# ==========================================================
# VIRTUAL CFD
# ==========================================================

def virtual_cfd(flow, w, h, l):

    w *= 1e-6
    h *= 1e-6
    l *= 1e-6

    Q = flow * 1e-9 / 60
    rho = 1000
    mu = 0.001

    A = w * h
    v = Q / A

    Dh = 2 * w * h / (w + h)

    Re = (rho * v * Dh) / mu
    dp = (32 * mu * v * l) / (Dh ** 2)
    tau = (4 * mu * v) / Dh

    return {
        "Velocity (m/s)": round(v, 6),
        "Re": round(Re, 2),
        "Pressure Drop (Pa)": round(dp, 2),
        "Shear (Pa)": round(tau, 4)
    }


# ==========================================================
# SINGLE RUN
# ==========================================================

if run_single:

    st.subheader("📊 Single Simulation")

    dqi, Re, _ = compute_dqi(flow, width, height, temp, material)

    psq, aff, res, lenf = compute_psq(
        dqi, flow, material, width, height, length
    )

    msq = compute_msq(material, temp, flow)

    fit = compute_fitness(dqi, psq, msq)

    df = pd.DataFrame(
        [[1, dqi, psq, msq, fit]],
        columns=["Run","DQI","PSQ","MSQ","Fitness"]
    )

    st.table(df)

    st.subheader("🧠 Optimization Plan")

    plan = optimization_engine(
        dqi, psq, msq, Re, aff, res, lenf, material
    )

    for p in plan:
        st.write("➡", p)


# ==========================================================
# BATCH
# ==========================================================

if run_batch:

    st.subheader("📊 Batch Experiments")

    data = []

    for i in range(25):

        f = flow * random.uniform(0.7,1.3)
        w = width * random.uniform(0.8,1.2)
        h = height * random.uniform(0.8,1.2)
        l = length * random.uniform(0.8,1.2)

        dqi,_,_ = compute_dqi(f,w,h,temp,material)
        psq,_,_,_ = compute_psq(dqi,f,material,w,h,l)
        msq = compute_msq(material,temp,f)
        fit = compute_fitness(dqi,psq,msq)

        data.append([i+1,dqi,psq,msq,fit])


    df = pd.DataFrame(
        data,
        columns=["Run","DQI","PSQ","MSQ","Fitness"]
    )

    st.dataframe(df)

    st.line_chart(df.set_index("Run")["Fitness"])


# ==========================================================
# GENETIC ALGORITHM
# ==========================================================

if run_ga:

    st.subheader("🧬 Genetic Optimization")

    pop = []

    for _ in range(20):
        pop.append([
            random.randint(60,250),
            random.randint(40,180),
            random.uniform(3,25)
        ])

    history = []

    for gen in range(12):

        scored = []

        for w,h,f in pop:

            dqi,_,_ = compute_dqi(f,w,h,temp,material)
            psq,_,_,_ = compute_psq(dqi,f,material,w,h,length)
            msq = compute_msq(material,temp,f)

            fit = compute_fitness(dqi,psq,msq)

            scored.append([w,h,f,fit])

        scored.sort(key=lambda x:x[3],reverse=True)

        history.append(scored[0][3])

        elites = scored[:10]

        pop = []

        for w,h,f,_ in elites:

            pop.append([
                w+random.randint(-10,10),
                h+random.randint(-8,8),
                f+random.uniform(-1,1)
            ])


    df = pd.DataFrame(
        {"Generation":range(1,13),"Fitness":history}
    )

    st.line_chart(df.set_index("Generation"))


# ==========================================================
# VIRTUAL CFD
# ==========================================================

if run_cfd:

    st.subheader("🌀 Virtual CFD")

    st.json(
        virtual_cfd(flow,width,height,length)
    )


# ==========================================================
# AUTO OPTIMIZATION
# ==========================================================

if run_auto:

    st.subheader("🤖 Automated Optimization")

    f = flow
    l = length
    m = material

    log = []

    for i in range(6):

        dqi,Re,_ = compute_dqi(f,width,height,temp,m)
        psq,aff,res,lenf = compute_psq(dqi,f,m,width,height,l)
        msq = compute_msq(m,temp,f)
        fit = compute_fitness(dqi,psq,msq)

        log.append([i+1,f,l,m,dqi,psq,msq,fit])

        plan = optimization_engine(
            dqi,psq,msq,Re,aff,res,lenf,m
        )

        f,l,m = apply_optimization(f,l,m,plan)


    df = pd.DataFrame(
        log,
        columns=[
            "Step","Flow","Length","Material",
            "DQI","PSQ","MSQ","Fitness"
        ]
    )

    st.dataframe(df)

    st.line_chart(df.set_index("Step")["PSQ"])


# ==========================================================
# LIVE MONITOR (SIMULATED)
# ==========================================================

if run_monitor:

    st.subheader("📡 Live Monitoring (Simulated)")

    placeholder = st.empty()

    log = []

    for i in range(40):

        f,t,p = simulate_sensors(flow,temp)

        dqi,_,_ = compute_dqi(f,width,height,t,material)
        psq,_,_,_ = compute_psq(dqi,f,material,width,height,length)
        msq = compute_msq(material,t,f)
        fit = compute_fitness(dqi,psq,msq)

        log.append([
            i+1,f,t,p,dqi,psq,msq,fit
        ])

        df = pd.DataFrame(
            log,
            columns=[
                "Time","Flow","Temp","Pressure",
                "DQI","PSQ","MSQ","Fitness"
            ]
        )

        placeholder.dataframe(df,use_container_width=True)

        if psq < 0.6:
            st.warning("⚠ Low Separation Efficiency")

        time.sleep(1)


    st.download_button(
        "Download Log",
        df.to_csv(index=False),
        "sensor_log.csv"
    )
