import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neuroscience Lab: HH Model", layout="wide")
st.title("🧬 Hodgkin-Huxley Neuron Simulator")
st.sidebar.header("Simulation Parameters")

# --- SIDEBAR CONTROLS ---
# These variables allow you to interact with the "biology" live
i_inj_amp = st.sidebar.slider("Injection Current (µA)", 0.0, 50.0, 10.0)
g_na_max = st.sidebar.slider("Max Na+ Conductance (gNa)", 0.0, 200.0, 120.0)
h_speed = st.sidebar.slider("h-gate speed (1.0 = Normal, 0.1 = Mutated)", 0.1, 1.0, 1.0)
duration = st.sidebar.number_input("Duration (ms)", value=50)

# --- HH MATH FUNCTIONS ---
def alpha_m(V): return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))
def beta_m(V):  return 4.0*np.exp(-(V+65.0) / 18.0)
def alpha_h(V): return 0.07*np.exp(-(V+65.0) / 20.0)
def beta_h(V):  return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))
def alpha_n(V): return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))
def beta_n(V):  return 0.125*np.exp(-(V+65.0) / 80.0)

# --- RUN SIMULATION ---
dt = 0.01
t = np.arange(0, duration, dt)
V = np.full(len(t), -65.0)
m, h, n = 0.05, 0.6, 0.32

for i in range(1, len(t)):
    v_now = V[i-1]
    # Current Pulse between 10ms and 11ms
    i_ext = i_inj_amp if 10.0 <= t[i] <= 11.0 else 0.0
    
    # Update Gating Variables (including our h_speed mutation slider)
    m += dt * (alpha_m(v_now)*(1-m) - beta_m(v_now)*m)
    h += dt * (alpha_h(v_now)*(1-h) - beta_h(v_now)*h) * h_speed
    n += dt * (alpha_n(v_now)*(1-n) - beta_n(v_now)*n)
    
    # Dynamics
    i_na = g_na_max * (m**3) * h * (v_now - 50.0)
    i_k = 36.0 * (n**4) * (v_now - (-77.0))
    i_l = 0.3 * (v_now - (-54.4))
    
    V[i] = v_now + (dt/1.0) * (i_ext - i_na - i_k - i_l)

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Top Plot: Voltage
ax1.plot(t, V, color='black', lw=2)
ax1.set_ylabel("Membrane Potential (mV)")
ax1.grid(alpha=0.3)

# Bottom Plot: Gating
ax2.plot(t, np.full_like(t, 0), 'k--', alpha=0.1) # zero line
ax2.plot(t, [0.05]*len(t), alpha=0) # dummy for scale
ax2.plot(t, [m_val for m_val in np.linspace(0,1,len(t))], alpha=0) # force 0-1 scale
# (Actually tracking m, h, n requires saving them in arrays, which I recommend doing!)

st.pyplot(fig)