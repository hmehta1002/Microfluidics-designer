# =====================================================
# AI MICROFLUIDIC OPTIMIZER
# DQI + PSQ + MSQ + GENETIC ALGORITHM
# =====================================================

import streamlit as st
import random
import pandas as pd
import math

# -----------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Microfluidic Optimizer",
    layout="wide"
)

st.title("🧬 AI-Based Microfluidic Optimization Platform")

st.markdown("""
Optimizes microfluidic designs for **protein purification**
using physics-based scoring and genetic algorithms.
""")

# -----------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------

st.sidebar.header("⚙️ Design Parameters")

geometry = st.sidebar.selectbox(
    "Geometry Type",
    ["Channel", "Serpentine", "Spiral"]
)

solver = st.sidebar.selectbox(
    "Solver (Virtual)",
    ["simpleFoam", "icoFoam"]
)

protein = st.sidebar.text_input(
    "Target Protein",
    "Albumin"
)

material = st.sidebar.selectbox(
    "Chip Material",
    ["PDMS", "PMMA", "Hydrogel", "Resin"]
)

flow_rate = st.sidebar.slider(
    "Flow Rate (µL/min)",
    1.0, 50.0, 10.0
)

width = st.sidebar.slider(
    "Channel Width (µm)",
    50, 300, 120
)

height = st.sidebar.slider(
    "Channel Height (µm)",
    30, 200, 60
)

length = st.sidebar.slider(
    "Channel Length (µm)",
    500, 3000, 1200
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    20, 60, 37
)

run_sim = st.sidebar.button("▶ Run Simulation")
run_ga = st.sidebar.button("🧬 Run Genetic Optimization")

# -----------------------------------------------------
# CORE MODELS
# -----------------------------------------------------

def run_simulation(flow, w, h, l, temp, material):

    w *= 1e-6
    h *= 1e-6

    Q = flow * 1e-9 / 60

    rho = 1000
    mu = 0.001

    A = w * h
    v = Q / A

    Dh = 2*w*h/(w+h)

    Re = (rho*v*Dh)/mu

    tau = (6*mu*Q)/(w*h**2)

    damage = math.exp(-0.015*tau)

    temp_factor = max(0.6, 1-abs(37-temp)/45)

    mat_factor = {
        "PDMS":0.85,
        "PMMA":0.75,
        "Hydrogel":0.9,
        "Resin":0.7
    }[material]

    laminar = min(1, 200/Re)

    dqi = (
        0.3*laminar +
        0.25*damage +
        0.2*temp_factor +
        0.25*mat_factor
    )

    return round(dqi,3), Re, tau, v


def compute_psq(dqi, flow, material, w, h):

    porosity = (w*h)/(500*500)

    affinity = {
        "PDMS":0.75,
        "PMMA":0.7,
        "Hydrogel":0.9,
        "Resin":0.65
    }[material]

    flow_factor = max(0.5, 1-flow/80)

    psq = (
        0.45*dqi +
        0.25*affinity +
        0.2*porosity +
        0.1*flow_factor
    )

    return round(psq,3)


def compute_msq(material, temp, flow):

    props = {

        "PDMS": [0.9,0.6,0.8,0.9,0.7],
        "PMMA": [0.75,0.7,0.85,0.8,0.75],
        "Hydrogel":[0.95,0.85,0.65,0.6,0.9],
        "Resin":[0.7,0.65,0.9,0.85,0.6]
    }

    b,a,s,f,w = props[material]

    temp_f = max(0.7,1-abs(37-temp)/60)
    flow_f = max(0.7,1-flow/120)

    affinity = 1-a

    msq = (
        0.25*b +
        0.20*affinity +
        0.20*s*temp_f +
        0.20*f +
        0.15*w*flow_f
    )

    return round(msq,3)


def fitness(dqi, psq, msq):

    return round(
        0.4*dqi + 0.35*psq + 0.25*msq,
        3
    )


# -----------------------------------------------------
# GENETIC ALGORITHM
# -----------------------------------------------------

def run_ga_engine(pop=20, gens=10):

    history = []
    best_all = None
    best_score = 0

    population = []

    for _ in range(pop):

        population.append({
            "w":random.randint(60,250),
            "h":random.randint(40,180),
            "l":random.randint(600,2500),
            "flow":random.uniform(3,25)
        })


    for g in range(gens):

        scored = []

        for ind in population:

            dqi,_,_,_ = run_simulation(
                ind["flow"],
                ind["w"],
                ind["h"],
                ind["l"],
                temperature,
                material
            )

            psq = compute_psq(
                dqi,
                ind["flow"],
                material,
                ind["w"],
                ind["h"]
            )

            msq = compute_msq(
                material,
                temperature,
                ind["flow"]
            )

            fit = fitness(dqi,psq,msq)

            scored.append((ind,fit))

            if fit > best_score:
                best_score = fit
                best_all = ind.copy()
                best_all["fitness"] = fit


        scored.sort(key=lambda x:x[1],reverse=True)

        history.append(scored[0][1])

        elites = scored[:pop//2]

        new_pop = []

        for ind,_ in elites:

            new_pop.append(ind)

            child = ind.copy()

            if random.random()<0.3:
                child["w"]+=random.randint(-15,15)

            if random.random()<0.3:
                child["h"]+=random.randint(-10,10)

            if random.random()<0.3:
                child["flow"]+=random.uniform(-2,2)

            new_pop.append(child)

        population = new_pop[:pop]


    return history, best_all


# -----------------------------------------------------
# SINGLE SIMULATION
# -----------------------------------------------------

if run_sim:

    st.subheader("📊 Simulation Results")

    dqi, Re, tau, v = run_simulation(
        flow_rate, width, height,
        length, temperature, material
    )

    psq = compute_psq(
        dqi, flow_rate,
        material, width, height
    )

    msq = compute_msq(
        material, temperature, flow_rate
    )

    fit = fitness(dqi,psq,msq)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("DQI", dqi)
    col2.metric("PSQ", psq)
    col3.metric("MSQ", msq)
    col4.metric("Fitness", fit)

    st.info(f"Re = {Re:.1f} | Shear = {tau:.3f} Pa | Velocity = {v:.4f} m/s")


# -----------------------------------------------------
# GENETIC OPTIMIZATION
# -----------------------------------------------------

if run_ga:

    st.subheader("🧬 Genetic Optimization")

    with st.spinner("Running optimization..."):

        history, best = run_ga_engine()

    df = pd.DataFrame({
        "Generation":range(1,len(history)+1),
        "Best Fitness":history
    })

    st.line_chart(df.set_index("Generation"))

    st.success("Optimization Complete")

    st.subheader("🏆 Best Design Found")

    st.json(best)

