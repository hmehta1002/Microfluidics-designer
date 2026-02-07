# =====================================================
# AI MICROFLUIDIC OPTIMIZER (WITH ADVISOR + MATERIAL AI)
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
Decision-support system for protein purification:
Simulation + AI + Optimization + Recommendations
""")


# -----------------------------------------------------
# TRAINABLE AI PARAMETERS
# -----------------------------------------------------

W = {
    "dqi": 0.4,
    "psq": 0.35,
    "msq": 0.25
}

BIAS = 0.1


# -----------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------

st.sidebar.header("⚙️ Controls")

protein = st.sidebar.text_input("Target Protein", "Albumin")

material = st.sidebar.selectbox(
    "Selected Material",
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
run_ga = st.sidebar.button("🧬 Run Optimization")
train_ai = st.sidebar.button("🧠 Train AI Model")
run_cfd = st.sidebar.button("🌀 Run Virtual CFD Tests")


# -----------------------------------------------------
# CORE PHYSICS MODEL (DQI)
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

    temp_factor = max(0.6,1-abs(37-temp)/45)

    mat_factor = {
        "PDMS":0.85,
        "PMMA":0.75,
        "Hydrogel":0.9,
        "Resin":0.7
    }[material]

    laminar = min(1,200/Re)

    dqi = (
        0.3*laminar +
        0.25*damage +
        0.2*temp_factor +
        0.25*mat_factor
    )

    return round(dqi,3), Re, tau, v


# -----------------------------------------------------
# VIRTUAL CFD ENGINE
# -----------------------------------------------------

def virtual_cfd(flow, w, h, l):

    w *= 1e-6
    h *= 1e-6
    l *= 1e-6

    Q = flow * 1e-9 / 60

    rho = 1000
    mu = 0.001

    A = w*h
    v = Q/A

    Dh = 2*w*h/(w+h)

    Re = (rho*v*Dh)/mu

    dp = (32*mu*v*l)/(Dh**2)

    tau = (4*mu*v)/Dh

    mixing = min(1,(Re/200)*(l/0.002))

    return {
        "Velocity (m/s)":round(v,5),
        "Reynolds":round(Re,2),
        "Pressure Drop (Pa)":round(dp,2),
        "Shear (Pa)":round(tau,4),
        "Mixing Efficiency":round(mixing,3)
    }


# -----------------------------------------------------
# PSQ MODEL
# -----------------------------------------------------

def compute_psq(dqi, flow, material, w, h):

    porosity = (w*h)/(500*500)

    affinity = {
        "PDMS":0.75,
        "PMMA":0.7,
        "Hydrogel":0.9,
        "Resin":0.65
    }[material]

    flow_factor = max(0.5,1-flow/80)

    psq = (
        0.45*dqi +
        0.25*affinity +
        0.2*porosity +
        0.1*flow_factor
    )

    return round(psq,3)


# -----------------------------------------------------
# MSQ MODEL
# -----------------------------------------------------

def compute_msq(material,temp,flow):

    props = {

        "PDMS":[0.9,0.6,0.8,0.9,0.7],
        "PMMA":[0.75,0.7,0.85,0.8,0.75],
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


# -----------------------------------------------------
# MATERIAL RECOMMENDER
# -----------------------------------------------------

def find_best_material(temp, flow):

    materials = ["PDMS", "PMMA", "Hydrogel", "Resin"]

    scores = {}

    for mat in materials:
        scores[mat] = compute_msq(mat, temp, flow)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked, ranked[0][0], ranked[0][1]


# -----------------------------------------------------
# DESIGN ADVISOR
# -----------------------------------------------------

def design_advisor(dqi, psq, msq, Re, tau, material, flow):

    tips = []

    if dqi < 0.7:
        if Re > 200:
            tips.append("Reduce flow rate to maintain laminar flow.")
        if tau > 2:
            tips.append("High shear stress: increase channel height.")
    else:
        tips.append("Flow dynamics are optimal.")

    if psq < 0.7:
        tips.append("Improve separation: increase length or reduce flow.")
    else:
        tips.append("Protein separation is efficient.")

    if msq < 0.7:
        tips.append("Material performance is low. Consider switching material.")
    else:
        tips.append("Selected material is suitable.")

    if material == "PDMS" and msq < 0.7:
        tips.append("PDMS causes adsorption. Hydrogel recommended.")

    if flow > 25:
        tips.append("Flow too high: may reduce binding efficiency.")

    if flow < 3:
        tips.append("Flow very low: throughput will be poor.")

    return tips


# -----------------------------------------------------
# AI FITNESS MODEL
# -----------------------------------------------------

def ai_fitness(dqi,psq,msq):

    return (
        W["dqi"]*dqi +
        W["psq"]*psq +
        W["msq"]*msq +
        BIAS
    )


# -----------------------------------------------------
# TRAINING SYSTEM
# -----------------------------------------------------

def generate_training_data(n=300):

    data = []

    for _ in range(n):

        w = random.randint(60,250)
        h = random.randint(40,180)
        l = random.randint(600,2500)
        flow = random.uniform(3,25)
        temp = random.randint(25,45)

        dqi,_,_,_ = run_simulation(
            flow,w,h,l,temp,material
        )

        psq = compute_psq(dqi,flow,material,w,h)
        msq = compute_msq(material,temp,flow)

        true = (
            0.45*dqi +
            0.30*psq +
            0.25*msq +
            random.uniform(-0.05,0.05)
        )

        data.append((dqi,psq,msq,true))

    return data


def train_model(data,lr=0.05,epochs=200):

    global W,BIAS

    for _ in range(epochs):

        dw = {"dqi":0,"psq":0,"msq":0}
        db = 0

        for dqi,psq,msq,y in data:

            err = ai_fitness(dqi,psq,msq)-y

            dw["dqi"]+=err*dqi
            dw["psq"]+=err*psq
            dw["msq"]+=err*msq
            db+=err

        n = len(data)

        for k in W:
            W[k]-=lr*dw[k]/n

        BIAS-=lr*db/n


# -----------------------------------------------------
# GENETIC ALGORITHM
# -----------------------------------------------------

def run_ga(pop=20,gens=10):

    population=[]

    for _ in range(pop):

        population.append({
            "w":random.randint(60,250),
            "h":random.randint(40,180),
            "l":random.randint(600,2500),
            "flow":random.uniform(3,25)
        })

    history=[]
    best=None
    best_score=0


    for _ in range(gens):

        scored=[]

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

            fit = ai_fitness(dqi,psq,msq)

            scored.append((ind,fit))

            if fit>best_score:
                best_score=fit
                best=ind.copy()
                best["fitness"]=round(fit,3)

        scored.sort(key=lambda x:x[1],reverse=True)

        history.append(scored[0][1])

        elites=scored[:pop//2]

        new_pop=[]

        for ind,_ in elites:

            new_pop.append(ind)

            child=ind.copy()

            if random.random()<0.3:
                child["w"]+=random.randint(-15,15)

            if random.random()<0.3:
                child["h"]+=random.randint(-10,10)

            if random.random()<0.3:
                child["flow"]+=random.uniform(-2,2)

            new_pop.append(child)

        population=new_pop[:pop]

    return history,best


# -----------------------------------------------------
# SINGLE SIMULATION
# -----------------------------------------------------

if run_sim:

    st.subheader("📊 Simulation Results")

    dqi,Re,tau,v = run_simulation(
        flow_rate,width,height,
        length,temperature,material
    )

    psq = compute_psq(
        dqi,flow_rate,
        material,width,height
    )

    msq = compute_msq(
        material,temperature,flow_rate
    )

    fit = ai_fitness(dqi,psq,msq)

    ranked,best_mat,best_score = find_best_material(
        temperature,flow_rate
    )

    advice = design_advisor(
        dqi,psq,msq,Re,tau,material,flow_rate
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("DQI",dqi)
    c2.metric("PSQ",psq)
    c3.metric("MSQ",msq)
    c4.metric("AI Fitness",round(fit,3))


    st.info(
        f"Re={Re:.1f} | Shear={tau:.4f} Pa | "
        f"Velocity={v:.5f} m/s"
    )


    # DESIGN REPORT
    st.subheader("🧠 Design Intelligence Report")

    for tip in advice:
        st.write("✔️",tip)


    # MATERIAL RECOMMENDATION
    st.subheader("🏗️ Material Recommendation")

    colA,colB = st.columns(2)

    colA.success(f"Best Material: {best_mat}")
    colA.metric("Best MSQ",best_score)

    rank_df = pd.DataFrame(
        ranked,
        columns=["Material","MSQ"]
    )

    colB.table(rank_df)


# -----------------------------------------------------
# TRAIN AI
# -----------------------------------------------------

if train_ai:

    st.subheader("🧠 AI Training")

    with st.spinner("Training..."):

        data = generate_training_data(400)
        train_model(data)

    st.success("Training Complete")

    st.write("Updated Weights")
    st.json(W)

    st.write("Bias:",round(BIAS,4))


# -----------------------------------------------------
# GENETIC OPTIMIZATION
# -----------------------------------------------------

if run_ga:

    st.subheader("🧬 Genetic Optimization")

    with st.spinner("Optimizing..."):

        history,best = run_ga()

    df = pd.DataFrame({
        "Generation":range(1,len(history)+1),
        "Best Fitness":history
    })

    st.line_chart(df.set_index("Generation"))

    st.success("Optimization Finished")

    st.subheader("🏆 Best Design")

    st.json(best)


# -----------------------------------------------------
# VIRTUAL CFD TESTS
# -----------------------------------------------------

if run_cfd:

    st.subheader("🌀 Virtual CFD Tests")

    results=[]

    for i in range(2):

        f = flow_rate*random.uniform(0.8,1.2)
        w = width*random.uniform(0.9,1.1)
        h = height*random.uniform(0.9,1.1)
        l = length*random.uniform(0.95,1.05)

        res = virtual_cfd(f,w,h,l)

        st.write(f"Test {i+1}")
        st.json(res)

        results.append(res)

    df = pd.DataFrame(results)

    st.dataframe(df)
