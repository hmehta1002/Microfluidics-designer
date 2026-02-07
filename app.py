# =====================================================
# AI MICROFLUIDIC OPTIMIZER (WITH LEARNING MODEL)
# DQI + PSQ + MSQ + GA + WEIGHTS + BIAS
# =====================================================

import streamlit as st
import random
import pandas as pd
import math

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Microfluidic Optimizer",
    layout="wide"
)

st.title("🧬 AI Microfluidic Optimization Platform")

st.markdown("""
Physics + Biotech + AI-based optimization
for protein purification systems.
""")

# -----------------------------------------------------
# AI MODEL PARAMETERS (TRAINABLE)
# -----------------------------------------------------

W = {
    "dqi": 0.4,
    "psq": 0.35,
    "msq": 0.25
}

BIAS = 0.1


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.header("⚙️ Controls")

protein = st.sidebar.text_input("Protein", "Albumin")

material = st.sidebar.selectbox(
    "Material",
    ["PDMS", "PMMA", "Hydrogel", "Resin"]
)

flow_rate = st.sidebar.slider(
    "Flow Rate (µL/min)",
    1.0, 50.0, 10.0
)

width = st.sidebar.slider(
    "Width (µm)",
    50, 300, 120
)

height = st.sidebar.slider(
    "Height (µm)",
    30, 200, 60
)

length = st.sidebar.slider(
    "Length (µm)",
    500, 3000, 1200
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    20, 60, 37
)

run_sim = st.sidebar.button("▶ Run Simulation")
run_ga = st.sidebar.button("🧬 Run Optimization")
train_ai = st.sidebar.button("🧠 Train AI Model")


# -----------------------------------------------------
# CORE MODELS
# -----------------------------------------------------

def run_simulation(flow, w, h, l, temp, material):

    # Convert units
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

    damage = math.exp(-0.015 * tau)

    temp_factor = max(0.6, 1 - abs(37 - temp) / 45)

    mat_factor = {
        "PDMS": 0.85,
        "PMMA": 0.75,
        "Hydrogel": 0.9,
        "Resin": 0.7
    }[material]

    laminar = min(1, 200 / Re)

    dqi = (
        0.3 * laminar +
        0.25 * damage +
        0.2 * temp_factor +
        0.25 * mat_factor
    )

    return round(dqi, 3), Re, tau, v


def compute_psq(dqi, flow, material, w, h):

    porosity = (w * h) / (500 * 500)

    affinity = {
        "PDMS": 0.75,
        "PMMA": 0.7,
        "Hydrogel": 0.9,
        "Resin": 0.65
    }[material]

    flow_factor = max(0.5, 1 - flow / 80)

    psq = (
        0.45 * dqi +
        0.25 * affinity +
        0.2 * porosity +
        0.1 * flow_factor
    )

    return round(psq, 3)


def compute_msq(material, temp, flow):

    props = {

        "PDMS": [0.9, 0.6, 0.8, 0.9, 0.7],
        "PMMA": [0.75, 0.7, 0.85, 0.8, 0.75],
        "Hydrogel": [0.95, 0.85, 0.65, 0.6, 0.9],
        "Resin": [0.7, 0.65, 0.9, 0.85, 0.6]
    }

    b, a, s, f, w = props[material]

    temp_f = max(0.7, 1 - abs(37 - temp) / 60)
    flow_f = max(0.7, 1 - flow / 120)

    affinity = 1 - a

    msq = (
        0.25 * b +
        0.20 * affinity +
        0.20 * s * temp_f +
        0.20 * f +
        0.15 * w * flow_f
    )

    return round(msq, 3)


# -----------------------------------------------------
# AI FITNESS MODEL
# -----------------------------------------------------

def ai_fitness(dqi, psq, msq):

    score = (
        W["dqi"] * dqi +
        W["psq"] * psq +
        W["msq"] * msq +
        BIAS
    )

    return score


# -----------------------------------------------------
# TRAINING SYSTEM
# -----------------------------------------------------

def generate_training_data(n=300):

    data = []

    for _ in range(n):

        w = random.randint(60, 250)
        h = random.randint(40, 180)
        l = random.randint(600, 2500)
        flow = random.uniform(3, 25)
        temp = random.randint(25, 45)

        dqi, _, _, _ = run_simulation(
            flow, w, h, l, temp, material
        )

        psq = compute_psq(dqi, flow, material, w, h)
        msq = compute_msq(material, temp, flow)

        true_score = (
            0.45 * dqi +
            0.30 * psq +
            0.25 * msq +
            random.uniform(-0.05, 0.05)
        )

        data.append((dqi, psq, msq, true_score))

    return data


def train_model(data, lr=0.05, epochs=200):

    global W, BIAS

    for _ in range(epochs):

        dw = {"dqi": 0, "psq": 0, "msq": 0}
        db = 0

        for dqi, psq, msq, y in data:

            y_pred = ai_fitness(dqi, psq, msq)

            err = y_pred - y

            dw["dqi"] += err * dqi
            dw["psq"] += err * psq
            dw["msq"] += err * msq
            db += err

        n = len(data)

        for k in W:
            W[k] -= lr * dw[k] / n

        BIAS -= lr * db / n


# -----------------------------------------------------
# GENETIC ALGORITHM
# -----------------------------------------------------

def run_ga(pop=20, gens=10):

    population = []

    for _ in range(pop):

        population.append({
            "w": random.randint(60, 250),
            "h": random.randint(40, 180),
            "l": random.randint(600, 2500),
            "flow": random.uniform(3, 25)
        })

    history = []
    best = None
    best_score = 0


    for g in range(gens):

        scored = []

        for ind in population:

            dqi, _, _, _ = run_simulation(
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

            fit = ai_fitness(dqi, psq, msq)

            scored.append((ind, fit))

            if fit > best_score:
                best_score = fit
                best = ind.copy()
                best["fitness"] = round(fit, 3)

        scored.sort(key=lambda x: x[1], reverse=True)

        history.append(scored[0][1])

        elites = scored[:pop // 2]

        new_pop = []

        for ind, _ in elites:

            new_pop.append(ind)

            child = ind.copy()

            if random.random() < 0.3:
                child["w"] += random.randint(-15, 15)

            if random.random() < 0.3:
                child["h"] += random.randint(-10, 10)

            if random.random() < 0.3:
                child["flow"] += random.uniform(-2, 2)

            new_pop.append(child)

        population = new_pop[:pop]


    return history, best


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

    fit = ai_fitness(dqi, psq, msq)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("DQI", round(dqi, 3))
    c2.metric("PSQ", round(psq, 3))
    c3.metric("MSQ", round(msq, 3))
    c4.metric("AI Fitness", round(fit, 3))

    st.info(
        f"Re = {Re:.1f} | "
        f"Shear = {tau:.3f} Pa | "
        f"Velocity = {v:.4f} m/s"
    )


# -----------------------------------------------------
# TRAIN AI
# -----------------------------------------------------

if train_ai:

    st.subheader("🧠 AI Training")

    with st.spinner("Training model..."):

        data = generate_training_data(400)
        train_model(data)

    st.success("Training Complete")

    st.write("Updated Weights")
    st.json(W)

    st.write("Bias:", round(BIAS, 4))


# -----------------------------------------------------
# GENETIC OPTIMIZATION
# -----------------------------------------------------

if run_ga:

    st.subheader("🧬 Genetic Optimization")

    with st.spinner("Optimizing..."):

        history, best = run_ga()

    df = pd.DataFrame({
        "Generation": range(1, len(history) + 1),
        "Best Fitness": history
    })

    st.line_chart(df.set_index("Generation"))

    st.success("Optimization Finished")

    st.subheader("🏆 Best Design Found")

    st.json(best)
