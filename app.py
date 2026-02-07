# =====================================================
# AI MICROFLUIDIC OPTIMIZER (CLEAN FINAL VERSION)
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

st.title("🧬 AI Microfluidic Optimization Platform")

st.markdown("""
Smart decision-support system for protein purification
using microfluidic devices.
""")


# -----------------------------------------------------
# AI WEIGHTS
# -----------------------------------------------------

W_DQI = 0.4
W_PSQ = 0.35
W_MSQ = 0.25
BIAS = 0.1


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.header("⚙️ Design Controls")

protein = st.sidebar.text_input("Target Protein", "Albumin")

material = st.sidebar.selectbox(
    "Selected Material",
    ["PDMS", "PMMA", "Hydrogel", "Resin"]
)

flow = st.sidebar.slider(
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

temp = st.sidebar.slider(
    "Temperature (°C)",
    20, 60, 37
)

run = st.sidebar.button("▶ Run Analysis")


# -----------------------------------------------------
# DQI MODEL
# -----------------------------------------------------

def compute_dqi(flow, w, h, temp, material):

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
# PSQ MODEL
# -----------------------------------------------------

def compute_psq(dqi, flow, material, w, h, l):

    porosity = (w*h)/(500*500)

    affinity = {
        "PDMS":0.75,
        "PMMA":0.7,
        "Hydrogel":0.9,
        "Resin":0.65
    }[material]

    flow_factor = max(0.5,1-flow/80)

    length_factor = min(1,l/2000)

    psq = (
        0.4*dqi +
        0.25*affinity +
        0.2*porosity +
        0.1*flow_factor +
        0.05*length_factor
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
# MATERIAL RANKING
# -----------------------------------------------------

def rank_materials(temp, flow):

    mats = ["PDMS","PMMA","Hydrogel","Resin"]

    scores = []

    for m in mats:
        scores.append((m,compute_msq(m,temp,flow)))

    scores.sort(key=lambda x:x[1],reverse=True)

    return scores


# -----------------------------------------------------
# AI FITNESS
# -----------------------------------------------------

def compute_fitness(dqi,psq,msq):

    return round(
        W_DQI*dqi +
        W_PSQ*psq +
        W_MSQ*msq +
        BIAS,
        3
    )


# -----------------------------------------------------
# ADVISOR
# -----------------------------------------------------

def advisor(dqi,psq,msq,Re,tau,material,flow,length):

    tips=[]

    # DQI
    if dqi<0.7:
        tips.append("Reduce flow to improve stability.")
        if tau>2:
            tips.append("Increase height to reduce shear.")
    else:
        tips.append("Flow dynamics are stable.")

    # PSQ
    if psq<0.6:
        tips.append("Poor separation: slow down flow.")
        tips.append("Increase channel length.")
        tips.append("Use Hydrogel for higher affinity.")

    elif psq<0.75:
        tips.append("Moderate separation: reduce flow slightly.")
        tips.append("Increase length by ~15%.")

    else:
        tips.append("Protein separation is high.")

    # MSQ
    if msq<0.7:
        tips.append("Material compatibility is low.")

        if material=="PDMS":
            tips.append("PDMS adsorbs proteins. Switch material.")

    else:
        tips.append("Material selection is good.")

    # Throughput
    if flow>25:
        tips.append("Flow too high: reduces PSQ.")

    if flow<3:
        tips.append("Flow too low: low productivity.")

    # Geometry
    if length<900:
        tips.append("Channel is short for good separation.")

    return tips


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

if run:

    st.subheader("📊 Performance Analysis")

    dqi,Re,tau,v = compute_dqi(
        flow,width,height,temp,material
    )

    psq = compute_psq(
        dqi,flow,material,width,height,length
    )

    msq = compute_msq(
        material,temp,flow
    )

    fitness = compute_fitness(dqi,psq,msq)

    materials = rank_materials(temp,flow)

    tips = advisor(
        dqi,psq,msq,Re,tau,
        material,flow,length
    )


    # Scores
    c1,c2,c3,c4 = st.columns(4)

    c1.metric("DQI",dqi)
    c2.metric("PSQ",psq)
    c3.metric("MSQ",msq)
    c4.metric("Fitness",fitness)


    st.info(
        f"Re = {Re:.1f} | "
        f"Shear = {tau:.3f} Pa | "
        f"Velocity = {v:.5f} m/s"
    )


    # Advisor
    st.subheader("🧠 Optimization Advice")

    for t in tips:
        st.write("✔️",t)


    # Materials
    st.subheader("🏗️ Material Recommendation")

    best_mat = materials[0][0]

    st.success(f"Recommended Material: {best_mat}")

    df = pd.DataFrame(
        materials,
        columns=["Material","MSQ"]
    )

    st.table(df)
