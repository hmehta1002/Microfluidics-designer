# ==========================================================
# COMPLETE AI MICROFLUIDIC OPTIMIZATION PLATFORM
# DQI + PSQ + MSQ + ANALYSIS + BATCH + GA + CFD + ADVISOR
# ==========================================================

import streamlit as st
import pandas as pd
import random
import math


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
• Hydrodynamic Analysis (DQI)
• Separation Analysis (PSQ)
• Material Analysis (MSQ)
• Batch Experiments
• Genetic Optimization
• Virtual CFD
• Design Advisor
""")


# ==========================================================
# AI WEIGHTS (JUSTIFIED)
# ==========================================================

W_DQI = 0.35
W_PSQ = 0.40   # Most important
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


# ==========================================================
# DQI: FLOW + MECHANICAL QUALITY
# ==========================================================

def compute_dqi(flow, w, h, temp, material):

    # Unit conversion
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

    return round(dqi, 3), Re, tau, v, laminar, shear_damage, temp_factor


# ==========================================================
# PSQ: BIOCHEMICAL SEPARATION QUALITY
# ==========================================================

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

    return round(psq, 3), affinity, residence, length_factor, porosity


# ==========================================================
# MSQ: MATERIAL PRACTICALITY
# ==========================================================

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

    return round(msq, 3), bio, affinity, stab, fab, wet


# ==========================================================
# FITNESS
# ==========================================================

def compute_fitness(dqi, psq, msq):

    return round(
        W_DQI * dqi +
        W_PSQ * psq +
        W_MSQ * msq +
        BIAS,
        3
    )


# ==========================================================
# DQI ANALYSIS
# ==========================================================

def analyze_dqi(dqi, Re, tau, lam, shear, tempf):

    rep = []

    if dqi > 0.8:
        rep.append("✅ Excellent flow stability.")
    elif dqi > 0.65:
        rep.append("⚠ Moderate flow quality.")
    else:
        rep.append("❌ Poor flow conditions.")

    if Re > 200:
        rep.append("⚠ Turbulence risk → Reduce flow.")

    if tau > 2:
        rep.append("⚠ High shear → Protein damage risk.")

    if tempf < 0.75:
        rep.append("⚠ Temperature deviation.")

    return rep


# ==========================================================
# PSQ ANALYSIS
# ==========================================================

def analyze_psq(psq, aff, res, lenf, por):

    rep = []

    if psq > 0.8:
        rep.append("✅ High purity separation.")
    elif psq > 0.65:
        rep.append("⚠ Moderate purity.")
    else:
        rep.append("❌ Low separation efficiency.")

    if aff < 0.7:
        rep.append("⚠ Weak surface binding.")

    if res < 0.7:
        rep.append("⚠ Flow too fast.")

    if lenf < 0.7:
        rep.append("⚠ Channel too short.")

    if por < 0.5:
        rep.append("⚠ Low surface area.")

    return rep


# ==========================================================
# MSQ ANALYSIS
# ==========================================================

def analyze_msq(msq, bio, aff, stab, fab, wet):

    rep = []

    if msq > 0.8:
        rep.append("✅ Excellent material choice.")
    elif msq > 0.65:
        rep.append("⚠ Acceptable material.")
    else:
        rep.append("❌ Poor material selection.")

    if bio < 0.8:
        rep.append("⚠ Low biocompatibility.")

    if fab < 0.7:
        rep.append("⚠ Hard to fabricate.")

    if wet < 0.7:
        rep.append("⚠ Poor wettability.")

    return rep


# ==========================================================
# MATERIAL RANKING
# ==========================================================

def rank_materials(temp, flow):

    mats = ["PDMS", "PMMA", "Hydrogel", "Resin"]

    res = []

    for m in mats:
        score, *_ = compute_msq(m, temp, flow)
        res.append((m, score))

    res.sort(key=lambda x: x[1], reverse=True)

    return res


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
# SINGLE SIMULATION
# ==========================================================

if run_single:

    st.subheader("📊 Single Simulation")

    dqi, Re, tau, v, lam, sh, tf = compute_dqi(
        flow, width, height, temp, material
    )

    psq, aff, res, lenf, por = compute_psq(
        dqi, flow, material, width, height, length
    )

    msq, bio, aff2, stab, fab, wet = compute_msq(
        material, temp, flow
    )

    fitness = compute_fitness(dqi, psq, msq)


    # MAIN TABLE
    df = pd.DataFrame(
        [[1, dqi, psq, msq, fitness]],
        columns=["Run_ID", "DQI", "PSQ", "MSQ", "Fitness"]
    )

    st.table(df)


    # ------------------ DQI ------------------

    st.subheader("🔵 DQI Analysis")

    for x in analyze_dqi(dqi, Re, tau, lam, sh, tf):
        st.write(x)


    # ------------------ PSQ ------------------

    st.subheader("🟢 PSQ Analysis")

    for x in analyze_psq(psq, aff, res, lenf, por):
        st.write(x)


    # ------------------ MSQ ------------------

    st.subheader("🟠 MSQ Analysis")

    for x in analyze_msq(msq, bio, aff2, stab, fab, wet):
        st.write(x)


    # ------------------ MATERIAL ------------------

    st.subheader("🏗️ Material Ranking")

    mats = rank_materials(temp, flow)

    st.success(f"Best Material: {mats[0][0]}")

    st.table(pd.DataFrame(mats, columns=["Material", "MSQ"]))


# ==========================================================
# BATCH SIMULATION
# ==========================================================

if run_batch:

    st.subheader("📊 Batch Simulation (25 Runs)")

    results = []

    for i in range(25):

        f = flow * random.uniform(0.7, 1.3)
        w = width * random.uniform(0.8, 1.2)
        h = height * random.uniform(0.8, 1.2)
        l = length * random.uniform(0.8, 1.2)

        dqi, _, _, _, _, _, _ = compute_dqi(
            f, w, h, temp, material
        )

        psq, *_ = compute_psq(dqi, f, material, w, h, l)

        msq, *_ = compute_msq(material, temp, f)

        fit = compute_fitness(dqi, psq, msq)

        results.append([
            i + 1, f, w, h, l, dqi, psq, msq, fit
        ])


    df = pd.DataFrame(
        results,
        columns=[
            "Run_ID", "Flow", "Width", "Height", "Length",
            "DQI", "PSQ", "MSQ", "Fitness"
        ]
    )

    st.dataframe(df)

    st.line_chart(df.set_index("Run_ID")["Fitness"])


# ==========================================================
# GENETIC ALGORITHM
# ==========================================================

if run_ga:

    st.subheader("🧬 Genetic Optimization")

    pop = []

    for i in range(20):
        pop.append([
            i + 1,
            random.randint(60, 250),
            random.randint(40, 180),
            random.uniform(3, 25)
        ])

    history = []

    for gen in range(12):

        scored = []

        for pid, w, h, f in pop:

            dqi, _, _, _, _, _, _ = compute_dqi(
                f, w, h, temp, material
            )

            psq, *_ = compute_psq(dqi, f, material, w, h, length)

            msq, *_ = compute_msq(material, temp, f)

            fit = compute_fitness(dqi, psq, msq)

            scored.append([pid, w, h, f, fit])


        scored.sort(key=lambda x: x[4], reverse=True)

        best = scored[0]

        history.append([gen + 1, best[0], best[4]])

        elites = scored[:10]

        new_pop = []

        for i, (_, w, h, f, _) in enumerate(elites):

            new_pop.append([
                i + 1,
                w + random.randint(-12, 12),
                h + random.randint(-10, 10),
                f + random.uniform(-1.5, 1.5)
            ])

        pop = new_pop


    df_hist = pd.DataFrame(
        history,
        columns=["Generation", "Best_ID", "Best_Fitness"]
    )

    st.table(df_hist)

    st.line_chart(df_hist.set_index("Generation")["Best_Fitness"])


# ==========================================================
# VIRTUAL CFD
# ==========================================================

if run_cfd:

    st.subheader("🌀 Virtual CFD Simulation")

    res = virtual_cfd(flow, width, height, length)

    st.json(res)
