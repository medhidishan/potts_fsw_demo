# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from potts_engine import run_simulation, MATERIALS
import imageio
import io

st.set_page_config(layout="wide", page_title="Potts Monte Carlo FSW demo")

st.title("Monte Carlo Potts — Microstructure / DRX Demo for FSW")

# Sidebar inputs
st.sidebar.header("Simulation inputs")

material = st.sidebar.selectbox("Material", list(MATERIALS.keys()))
L = st.sidebar.slider("Grid size (L x L)", min_value=50, max_value=400, value=200, step=25)
n_init = st.sidebar.slider("Initial grain count", 10, 500, 100, step=10)
T_global = st.sidebar.slider("Global Temperature (K)", 300, 1500, 900, step=50)
strain = st.sidebar.number_input("Total engineering strain (e.g., 0.02)", min_value=0.0, value=0.02, step=0.01, format="%.4f")
strain_rate = st.sidebar.number_input("Strain rate (1/s)", min_value=0.0, value=1.0, step=0.1)
mcs_steps = st.sidebar.slider("MCS steps", 10, 2000, 500, step=10)
snapshot_every = st.sidebar.slider("Snapshot every (MCS)", 1, 200, 50)
use_numba = st.sidebar.checkbox("Use numba (if installed)", value=False)

st.sidebar.markdown("**Physics options**")
J = st.sidebar.number_input("Boundary energy J (heuristic)", value=1.0, step=0.1)
run_button = st.sidebar.button("Run simulation")

st.sidebar.markdown("**Outputs**")
st.sidebar.write("Snapshots will be shown below. Use download buttons for images & CSV.")

# Run or show instructions
if not run_button:
    st.info("Set inputs in the left panel and click **Run simulation** to start. (This prototype runs on CPU.)")
    st.stop()

# Run simulation (blocking)
with st.spinner("Running simulation (this may take a few minutes depending on grid & MCS)..."):
    result = run_simulation(Lx=L, Ly=L, n_init=n_init, J=J, mcs_steps=mcs_steps,
                            T_global=T_global, strain=strain, strain_rate=strain_rate,
                            material_name=material, use_numba=use_numba, snapshot_every=snapshot_every)

if result.get("status") == "fatigue":
    st.error("Input strain exceeds allowable for this material — simulation halted (fatigue condition).")
    st.write(f"Allowable strain (heuristic) = {result['allowable_strain']:.6f}")
    st.write("Fatigue mask (locations where local strain > allowable):")
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(result["fatigued_mask"].T, origin="lower")
    ax.set_title("Fatigued locations")
    ax.axis("off")
    st.pyplot(fig)
    st.stop()

# Show snapshots
snapshots = result["snapshots"]
stats = pd.DataFrame(result["stats"])
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Microstructure snapshots")
    # show latest snapshot with a slider to choose snapshot
    mcs_list = [m for m,_,_ in snapshots]
    idx = st.select_slider("Choose snapshot (MCS)", options=mcs_list, value=mcs_list[-1])
    sel = [s for s in snapshots if s[0]==idx][0]
    _, grid_img, rho = sel
    # colorize grid by label mapped to colormap (mod by 256)
    cmap = plt.cm.get_cmap("tab20")
    normed = (grid_img % 20)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(normed.T, origin="lower")
    ax.set_title(f"Microstructure at MCS={idx}")
    ax.axis("off")
    st.pyplot(fig)

    # downloadable PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download snapshot PNG", buf, file_name=f"snapshot_mcs_{idx}.png")

with col2:
    st.subheader("Statistics")
    st.write(stats.tail(10))
    # plot mean_area vs mcs
    fig2, ax2 = plt.subplots()
    ax2.plot(stats['mcs'], stats['mean_area'], '-o')
    ax2.set_xlabel("MCS")
    ax2.set_ylabel("Mean grain area")
    ax2.grid(True)
    st.pyplot(fig2)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    buf2.seek(0)
    st.download_button("Download mean area plot", buf2, file_name="mean_area_vs_mcs.png")

st.subheader("Dislocation density (rho) map for selected snapshot")
fig3, ax3 = plt.subplots(figsize=(4,4))
ax3.imshow(rho.T, origin="lower")
ax3.set_title("rho map")
ax3.axis("off")
st.pyplot(fig3)
buf3 = io.BytesIO()
fig3.savefig(buf3, format="png", bbox_inches="tight")
buf3.seek(0)
st.download_button("Download rho map", buf3, file_name=f"rho_snapshot_{idx}.png")

st.markdown("---")
st.write("**Download simulation stats (CSV)**")
csv_buf = io.BytesIO()
stats.to_csv(csv_buf, index=False)
csv_buf.seek(0)
st.download_button("Download stats.csv", csv_buf, file_name="simulation_stats.csv")

#dishan