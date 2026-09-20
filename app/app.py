"""
Streamlit demo: predict remaining useful life (RUL) for a turbofan engine
from its recent sensor history.

Run with:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make ../src importable regardless of where streamlit is launched from
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import ALL_COLUMNS, FEATURE_COLUMNS, WINDOW_SIZE  # noqa: E402
from predict import RULPredictor  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_engines"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="C-MAPSS RUL Predictor", page_icon="\u2708\ufe0f", layout="centered")


@st.cache_resource
def get_predictor():
    return RULPredictor(models_dir=str(MODELS_DIR))


@st.cache_data
def load_sample_engines():
    with open(SAMPLE_DIR / "true_rul.json") as f:
        true_rul = json.load(f)
    engines = {}
    for unit_str, rul in true_rul.items():
        csv_path = SAMPLE_DIR / f"engine_{unit_str}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            # Fallback: zlib+base64 sidecar when binary/large CSV upload is blocked
            import base64, zlib, io
            z64_path = SAMPLE_DIR / f"engine_{unit_str}.csv.z64"
            df = pd.read_csv(io.BytesIO(zlib.decompress(base64.b64decode(z64_path.read_text()))))
        engines[int(unit_str)] = {"df": df, "true_rul": rul}
    return engines


@st.cache_data
def load_test_predictions():
    """Per-engine (true, predicted) RUL for all 100 official test engines, written by train.py."""
    path = MODELS_DIR / "test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Accept either a headerless raw C-MAPSS file or a CSV with matching column names."""
    raw = pd.read_csv(uploaded_file)
    # If the columns don't look like our expected names, assume it's headerless
    # (i.e. the "header" pandas picked up is actually the first data row).
    if not set(["unit_number", "time_cycles"]).issubset(raw.columns):
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, sep=r"\s+|,", header=None, names=ALL_COLUMNS, engine="python")
    return raw


st.title("\u2708\ufe0f Turbofan Remaining Useful Life (RUL) Predictor")
st.caption(
    "Doubler Aerospace \u2014 Phase 01 ground-side software prototype. "
    "Trained on NASA's simulated C-MAPSS FD001 benchmark."
)
st.warning(
    "**Simulated NASA benchmark data. Not for flight decisions.** "
    "This is a research/portfolio demo, not certified prognostics software.",
    icon="\u26a0\ufe0f",
)

tab_predict, tab_performance = st.tabs(["Predict an engine", "Model performance"])

# ============================================================================
# TAB 1: single-engine prediction (sample engine or upload your own)
# ============================================================================
with tab_predict:
    source = st.radio("Data source", ["Bundled sample engine", "Upload your own CSV"], horizontal=True)

    engine_df = None
    true_rul = None
    engine_label = None

    if source == "Bundled sample engine":
        samples = load_sample_engines()
        unit = st.selectbox(
            "Pick a sample test engine",
            options=sorted(samples.keys()),
            format_func=lambda u: f"Engine {u}  (true RUL = {samples[u]['true_rul']} cycles)",
        )
        engine_df = samples[unit]["df"]
        true_rul = samples[unit]["true_rul"]
        engine_label = f"Engine {unit}"
    else:
        uploaded = st.file_uploader(
            f"Upload a CSV with at least {WINDOW_SIZE} rows of one engine's cycles "
            "(the 26 standard C-MAPSS columns, with or without a header row).",
            type=["csv", "txt"],
        )
        if uploaded is not None:
            try:
                engine_df = parse_uploaded_csv(uploaded)
                engine_label = "Uploaded engine"
            except Exception as e:
                st.error(f"Couldn't parse that file: {e}")

    if engine_df is not None:
        n_cycles = len(engine_df)
        if n_cycles < WINDOW_SIZE:
            st.error(f"This engine only has {n_cycles} cycles of data; need at least {WINDOW_SIZE}.")
        else:
            st.subheader(engine_label)

            # Let the user "rewind" the engine to see how the prediction changes over its life
            cutoff = st.slider(
                "Cycles of history to feed the model (simulates checking in at an earlier point in the engine's life)",
                min_value=WINDOW_SIZE,
                max_value=n_cycles,
                value=n_cycles,
                step=1,
            )
            visible_df = engine_df[engine_df["time_cycles"] <= engine_df["time_cycles"].min() + cutoff - 1]

            predictor = get_predictor()
            try:
                pred = predictor.predict_one(visible_df)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred = None

            if pred is not None:
                cols = st.columns(3) if true_rul is not None else st.columns(1)
                cols[0].metric("Predicted RUL", f"{pred:.0f} cycles")
                if true_rul is not None:
                    cols[1].metric("True RUL (at full trajectory)", f"{true_rul} cycles")
                    cols[2].metric("Error", f"{pred - true_rul:+.0f} cycles")

            st.markdown("**Sensor trends (this engine, cycles shown so far):**")
            sensor_options = [c for c in engine_df.columns if c.startswith("sensor_")]
            default_sensors = [s for s in ["sensor_11", "sensor_4", "sensor_9"] if s in sensor_options][:3]
            chosen_sensors = st.multiselect("Sensors to plot", sensor_options, default=default_sensors)
            if chosen_sensors:
                raw_chart_df = visible_df.set_index("time_cycles")[chosen_sensors]
                normalize = st.checkbox(
                    "Normalize each sensor to 0-1",
                    value=True,
                    help=(
                        "Sensors are recorded in very different units and ranges "
                        "(e.g. one sensor sits around 9,000, another around 47). "
                        "Plotted raw on the same axis, small-range sensors look "
                        "flat even when they're trending. Normalizing rescales "
                        "each sensor to its own min/max so trends are visible "
                        "side by side. This only affects the chart -- the model's "
                        "predictions always use properly scaled data internally."
                    ),
                )
                if normalize:
                    col_min = raw_chart_df.min()
                    col_max = raw_chart_df.max()
                    span = (col_max - col_min).replace(0, 1)  # avoid divide-by-zero for a flat column
                    chart_df = (raw_chart_df - col_min) / span
                    st.caption("Each line rescaled to its own 0-1 range for visual comparison.")
                else:
                    chart_df = raw_chart_df
                st.line_chart(chart_df)

            with st.expander("How this prediction was made"):
                st.markdown(
                    f"""
                    1. The last **{WINDOW_SIZE} cycles** of this engine's history were taken.
                    2. For each of the **{len(FEATURE_COLUMNS)} informative sensors/settings**
                       (7 near-constant channels were dropped), six summary statistics were
                       computed (mean, std, min, max, last value, trend/slope).
                    3. Those **{len(FEATURE_COLUMNS) * 6} features** were scaled with the same
                       `StandardScaler` fit during training, then passed to a gradient-boosted
                       tree model (XGBoost) trained on 80 engines from the FD001 training set.
                    4. The model outputs a single number: estimated cycles remaining before
                       this engine would reach the failure threshold, as defined by the
                       simulation.
                    """
                )
    else:
        st.info("Pick a sample engine above, or upload your own CSV, to get a prediction.")

# ============================================================================
# TAB 2: model performance -- predicted vs. true RUL on all 100 test engines
# ============================================================================
with tab_performance:
    st.subheader("Predicted vs. true RUL \u2014 all 100 official test engines")
    st.caption(
        "Every dot is a real test engine, scored the official way: the model's "
        "prediction from its last 30 cycles, compared against the true remaining "
        "life it never saw during training. The dashed line is where a perfect "
        "model would put every point \u2014 the closer a dot sits to it, the better "
        "that call was."
    )

    perf_df = load_test_predictions()
    metrics = load_metrics()

    if perf_df is None or metrics is None:
        st.info(
            "No saved performance data found yet. Run `python train.py` from the "
            "`src/` folder first \u2014 it writes `models/test_predictions.csv` and "
            "`models/metrics.json`, which is what this tab reads."
        )
    else:
        axis_max = max(perf_df["true_rul"].max(), perf_df["predicted_rul"].max()) + 5

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, axis_max], [0, axis_max], linestyle="--", color="gray", linewidth=1, label="Perfect prediction")
        ax.scatter(
            perf_df["true_rul"],
            perf_df["predicted_rul"],
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
            s=35,
        )
        ax.set_xlabel("True RUL (cycles)")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("XGBoost: Predicted vs. True RUL")
        st.pyplot(fig)
        plt.close(fig)  # Streamlit reruns the whole script on every interaction anywhere
        # in the app (not just this tab) -- without closing it, matplotlib keeps every
        # old figure in memory forever, which slowly leaks RAM on a long-running deployed app.

        xgb_metrics = metrics["xgboost"]["test"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Test RMSE", f"{xgb_metrics['rmse']:.1f} cycles")
        col2.metric("Test NASA score", f"{xgb_metrics['nasa_score']:.1f}")
        col3.metric("Test engines", f"{xgb_metrics['n']}")

        with st.expander("Compare all 4 models trained"):
            rows = []
            for name, r in metrics.items():
                rows.append(
                    {
                        "Model": name,
                        "Val RMSE": round(r["val"]["rmse"], 2),
                        "Val NASA score": round(r["val"]["nasa_score"], 1),
                        "Test RMSE": round(r["test"]["rmse"], 2),
                        "Test NASA score": round(r["test"]["nasa_score"], 1),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                "Each model beating the previous one, in this order, on data none of "
                "them trained on, is the main evidence this pipeline is sound rather "
                "than lucky."
            )

        with st.expander("Worst 5 individual predictions"):
            worst = perf_df.reindex(perf_df["error"].abs().sort_values(ascending=False).index).head(5)
            st.dataframe(
                worst[["unit_number", "true_rul", "predicted_rul", "error"]].round(1),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Shown for transparency \u2014 these aren't hidden. The middle of an "
                "engine's life is inherently harder to call precisely than its very "
                "start or end."
            )
