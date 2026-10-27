"""EV Charge Time Predictor: Streamlit version.

Run:  streamlit run streamlit_app.py
"""
import altair as alt
import pandas as pd
import streamlit as st

import ml
from simulator import CHARGERS

LR_COLOR, DT_COLOR = "#2A62C4", "#C2600A"
CHARGER_OPTIONS = {f"{p} kW {'DC' if p > 22 else 'AC'}": p for p in CHARGERS}

st.set_page_config(page_title="EV Charge Time Predictor", layout="wide")


def fmt_time(minutes):
    minutes = max(0, round(minutes))
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


st.title("EV Charge Time Predictor")
st.write("Linear Regression and a Decision Tree, both trained with scikit-learn. "
         "Set up a charging session and compare what each model predicts.")

left, right = st.columns([1, 2], gap="large")

# ---------------- inputs ----------------
with left:
    st.subheader("Charging session")
    battery = st.slider("Battery capacity (kWh)", 25, 100, 40)
    start, target = st.slider("Charge from % to %", 0, 100, (20, 80))
    st.caption("Cars charge more slowly above 80%.")
    charger_label = st.selectbox("Charger", list(CHARGER_OPTIONS), index=4)
    temp = st.slider("Outside temperature (°C)", -5, 45, 28)
    with st.expander("Decision Tree settings"):
        depth = st.slider("Maximum depth", 1, 12, 8)
        leaf = st.slider("Minimum sessions per leaf", 1, 40, 5)

if target - start < 5:
    st.warning("Pick a charge range of at least 5 percentage points.")
    st.stop()

charger = CHARGER_OPTIONS[charger_label]
res = ml.predict(battery, start, target, charger, temp, depth, leaf)
summary = ml.summary(depth, leaf)

# ---------------- predictions ----------------
with right:
    st.subheader("Predicted charging time")
    c1, c2 = st.columns(2)
    sim = res["sim_min"]
    for col, name, minutes in ((c1, "Linear Regression", res["lr_min"]), (c2, "Decision Tree", res["dt_min"])):
        delta = f"{minutes - sim:+.0f} min vs simulator" if sim is not None else None
        col.metric(name, fmt_time(minutes), delta, delta_color="off")
    if sim is not None:
        st.caption(f"Simulator reference: **{fmt_time(sim)}**. This is what the charging rules give without random noise.")
    if res["beyond_range"]:
        st.warning("This session takes longer than the 12 hours in the training data, "
                   "so both models are guessing beyond what they saw.")

    with st.expander("Why the tree predicted this", expanded=True):
        for step in res["path"]:
            threshold = f"{step['threshold']:.1f}".rstrip("0").rstrip(".")
            st.write(f"{step['label']} ≤ {threshold} {step['unit']}?  **{'yes' if step['left'] else 'no'}**")
        st.write(f"Leaf: average of {res['leaf_samples']} training sessions = **{fmt_time(res['leaf_value'])}**")

# ---------------- accuracy ----------------
st.divider()
st.subheader("How accurate are they?")
st.caption(f"Scored on {summary['n_test']} sessions the models did not see while learning. "
           f"They learned from {summary['n_train']}.")
lr, dt = summary["lr"], summary["dt"]
table = pd.DataFrame({
    "Score": ["Mean absolute error", "Root mean squared error", "R² on unseen sessions", "R² on training sessions"],
    "Linear Regression": [f"{lr['mae']:.1f} min", f"{lr['rmse']:.1f} min", f"{lr['r2_test']:.3f}", f"{lr['r2_train']:.3f}"],
    "Decision Tree": [f"{dt['mae']:.1f} min", f"{dt['rmse']:.1f} min", f"{dt['r2_test']:.3f}", f"{dt['r2_train']:.3f}"],
}).set_index("Score")
st.table(table)

tree_wins = dt["mae"] < lr["mae"]
winner, loser = ("The Decision Tree", "Linear Regression") if tree_wins else ("Linear Regression", "the Decision Tree")
w_mae, l_mae = (dt["mae"], lr["mae"]) if tree_wins else (lr["mae"], dt["mae"])
verdict = (f"{winner} predicts better on unseen sessions. It is off by {w_mae:.1f} min on average, "
           f"against {l_mae:.1f} min for {loser}.")
if dt["r2_train"] - dt["r2_test"] > 0.03:
    verdict += (" The tree scores higher on its training sessions than on unseen ones, which is overfitting. "
                "A smaller depth narrows the gap.")
st.write(verdict)

# ---------------- chart ----------------
n = len(summary["actual"])
points = pd.DataFrame({
    "Actual": summary["actual"] * 2,
    "Predicted": summary["lr_pred"] + summary["dt_pred"],
    "Model": ["Linear Regression"] * n + ["Decision Tree"] * n,
})
top = float(max(points["Actual"].max(), points["Predicted"].max()))
colors = alt.Scale(domain=["Linear Regression", "Decision Tree"], range=[LR_COLOR, DT_COLOR])

dots = alt.Chart(points).mark_circle(size=40, opacity=0.5).encode(
    x=alt.X("Actual:Q", title="Actual charging time (min)"),
    y=alt.Y("Predicted:Q", title="Predicted (min)"),
    color=alt.Color("Model:N", scale=colors),
    tooltip=["Model", "Actual", "Predicted"],
)
diagonal = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(
    strokeDash=[5, 4], color="gray").encode(x="x:Q", y="y:Q")
layers = diagonal + dots

if sim is not None:
    yours = pd.DataFrame({
        "Actual": [sim, sim],
        "Predicted": [res["lr_min"], res["dt_min"]],
        "Model": ["Linear Regression", "Decision Tree"],
    })
    layers = layers + alt.Chart(yours).mark_point(shape="circle", size=220, filled=False, strokeWidth=3).encode(
        x="Actual:Q", y="Predicted:Q", color=alt.Color("Model:N", scale=colors, legend=None))

st.subheader("Predicted vs actual")
st.caption("Each dot is a test session. Dots on the dashed line are perfect predictions. "
           "The large rings are your session above.")
st.altair_chart(layers.properties(height=380, width="container"))

# ---------------- what each model learned ----------------
st.divider()
st.subheader("What each model learned")
a, b = st.columns(2, gap="large")
with a:
    st.markdown("**Linear Regression**")
    st.caption("Effect on the minutes of a typical change in each input.")
    coef = pd.DataFrame(summary["lr_coef"]).set_index("label")
    st.bar_chart(coef[["standardized"]].rename(columns={"standardized": "Minutes per typical change"}), color=LR_COLOR)
    st.dataframe(coef[["per_unit", "unit"]].rename(columns={"per_unit": "min per unit"}))
with b:
    st.markdown("**Decision Tree**")
    st.caption(f"{summary['dt_leaves']} leaves, {summary['dt_depth']} levels deep. "
               "Bars show how much each input helped split the sessions.")
    imp = pd.DataFrame(summary["dt_importance"]).set_index("label")
    st.bar_chart(imp.rename(columns={"value": "Share of splitting power"}), color=DT_COLOR)

st.divider()
if summary["simulated"]:
    st.caption("About the data: the sessions in data/ev_sessions.csv are generated by simulator.py from simple charging "
               "rules (energy to add, charger and car limits, slower charging above 80%, hot and cold weather, random "
               "noise). They are simulated, not measured from real vehicles.")
else:
    st.caption(f"The models are trained on {summary['data_file']} ({summary['n_train'] + summary['n_test']} rows).")
