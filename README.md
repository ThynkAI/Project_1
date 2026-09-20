# C-MAPSS Remaining Useful Life (RUL) Predictor

**Doubler Aerospace — Phase 01 ground-side software prototype (Project 1 of 5)**

Predicts how many operating cycles a simulated turbofan engine has left before
failure, from its recent sensor history. Trained and scored on NASA's C-MAPSS
FD001 benchmark.

> **Simulated NASA benchmark data. Not for flight decisions.** This is a
> research/portfolio demo, not certified prognostics software, and makes no
> airworthiness claims.

---

## 1. What's in this folder

```
rul_project/
├── data/                    # raw C-MAPSS FD001 files (not committed if private)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── src/
│   ├── config.py            # every constant the pipeline uses, in one place
│   ├── data_prep.py         # load raw files, label RUL, drop dead columns
│   ├── features.py          # sliding-window feature engineering
│   ├── metrics.py           # RMSE + NASA PHM-2008 scoring function
│   ├── train.py             # trains all 4 models, saves the best one
│   └── predict.py           # loads the saved model, predicts on new data
├── models/                  # produced by train.py
│   ├── xgb_model.joblib
│   ├── scaler.joblib
│   ├── feature_names.json
│   ├── metrics.json
│   └── config_snapshot.json
├── app/
│   ├── app.py                # Streamlit demo
│   └── sample_engines/       # 5 bundled test engines so the app works with no upload
├── requirements.txt
└── README.md                 # this file
```

## 2. The data

**Source:** NASA's Prognostics Center of Excellence (PCoE), via the C-MAPSS
(Commercial Modular Aero-Propulsion System Simulation) turbofan degradation
simulator. Reference: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
*Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*,
1st International Conference on Prognostics and Health Management (PHM08).

**A heads-up if you're getting the data yourself:** NASA's own data.nasa.gov
listing currently shows C-MAPSS as unavailable for direct download ("Glenn
Research Center management is reviewing the availability requirements").
The dataset is still freely available — same files, CC0 public domain — via
Kaggle's "NASA Turbofan Jet Engine Data Set" mirror, which is what was used
here.

**FD001 subset specifically:**
- 100 training engines, run all the way to failure
- 100 test engines, each cut off at a random point *before* failure
- One operating condition (sea level), one fault mode (HPC degradation) —
  the simplest of the four C-MAPSS subsets, which is why it ships first
- 26 raw columns per row: unit number, cycle number, 3 operational settings,
  21 sensor readings
- Every engine (train and test) has at least 31 cycles, confirmed before
  building the windowing logic — that matters because it means every engine
  can support a full 30-cycle window with zero padding.

## 3. Method, step by step

This mirrors the order the pipeline actually runs in.

### 3.1 Labeling (`data_prep.py`)

Training engines run to failure, so at any row we know exactly how many
cycles are left: `(that engine's final cycle) − (this row's cycle)`.

That raw countdown is then **clipped at 125 cycles**:
`RUL = min(max_cycle − cycle, 125)`. Early in an engine's life, degradation
isn't reliably visible in the sensors — the difference between "290 cycles
left" and "250 cycles left" looks like noise to the model, not signal. Clipping
treats early life as a flat, healthy plateau and only asks the model to track
RUL once it's within the clip window, which is where the sensors actually
carry information.

### 3.2 Dropping dead sensors (`data_prep.py`)

Checked `std()` and `nunique()` for every setting/sensor column on the
training data. Seven columns came back with exactly **one unique value**
across all 20,631 rows — zero variance, zero signal:

`setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19`

These are dropped before anything else touches the data. That leaves
**17 feature columns** (2 settings + 15 sensors) that actually vary.

### 3.3 Windowing (`features.py`)

Tree models and linear models take a flat feature vector, not a sequence —
so instead of feeding raw per-cycle rows, each training example is a
**30-cycle sliding window**, summarized into six statistics per feature
column:

- `mean`, `std`, `min`, `max` — the window's overall level and spread
- `last` — the most recent reading (where the engine is *right now*)
- `slope` — a least-squares linear trend across the window (is this sensor
  actively degrading, or flat?)

17 columns × 6 stats = **102 engineered features** per window. The label for
a window is the RUL at the window's *last* cycle.

For **training and validation**, every possible window (stride 1) is built
for each engine — an engine with 200 cycles yields 171 overlapping windows,
so the model sees engines at every stage of life, not just the end.

For the **official test score**, only the *last* 30 cycles of each test
engine are used — one window per engine — because that mirrors the real
question the benchmark asks: "given what you've seen of this engine so far,
how much life is left right now?"

### 3.4 Train/validation split (`config.py`, `train.py`)

Split **by engine ID**, not by row: engines 1–80 → training, engines 81–100
→ validation. A random row-level split would let cycles from the *same*
engine's failure trajectory land in both train and validation — the model
could then partly memorize that engine's specific decay curve instead of
learning to generalize to one it's never seen. Splitting by whole engine
avoids that leak entirely.

### 3.5 Scaling (`train.py`)

A single `StandardScaler` is **fit only on the 13,818 training windows**
(engines 1–80), then used to transform validation and test features. Fitting
on anything outside the training engines — even just to compute a mean/std —
would leak information about engines the model is supposed to have never
seen.

### 3.6 Models, in order of sophistication (`train.py`)

| Step | Model | Why it's here |
|---|---|---|
| 1 | Mean baseline | Predicts the average training RUL for everything — the floor any real model has to beat |
| 2 | Ridge regression | Simple linear model on the scaled window stats |
| 3 | Random Forest | Non-linear, handles feature interactions, still fast to train |
| 4 | **XGBoost** (headline) | Gradient-boosted trees — the model actually saved and shipped in the app |

An LSTM (which would consume the raw 30-timestep sequence directly instead
of summary stats) is a natural next step, but only after XGBoost was saved
and working, per the project's own operating order — not required to ship.

## 4. Results

Two numbers per model, on two different sets:

- **Validation** — 3,913 windows from the 20 held-out training engines
  (81–100), all points in their life, not just the end
- **Test** — the actual 100 C-MAPSS test engines, scored the official way
  (last window per engine vs. the true RUL in `RUL_FD001.txt`)

| Model | Val RMSE | Val NASA score | **Test RMSE** | **Test NASA score** |
|---|---:|---:|---:|---:|
| Mean baseline | 42.18 | 644,731.3 | 41.74 | 16,938.6 |
| Ridge | 17.30 | 16,450.8 | 17.51 | 452.1 |
| Random Forest | 14.47 | 13,087.9 | 14.53 | 318.9 |
| **XGBoost** | **13.70** | **10,336.1** | **14.31** | **300.1** |

**Read the trend, not just the last row:** each model beats the one before
it, on both val and test, in the same order — that consistency is what says
the pipeline is sound rather than lucky. XGBoost's test RMSE of 14.3 cycles
sits right in the "low-teens to mid-20s" range that's expected for a
correctly-built FD001 pipeline; a result in the 50s–80s would have meant a
leak or a labeling bug somewhere upstream, and a result noticeably better
than ~12 would suggest test-label leakage instead — `train.py` checks both
ends automatically and prints a warning if either happens.

### 4.1 Against published baselines

The number above only means something next to other papers that scored the
*same* FD001 test split the *same* way (last window per engine vs.
`RUL_FD001.txt`) — see Ramasso & Saxena (2014), who warn that PHM'08
challenge-data scores and public FD001–FD004 scores are not the same
leaderboard. With that caveat, our number sits alongside:

| Source | Model | Test RMSE | Test NASA score |
|---|---|---:|---:|
| **This project** | **XGBoost, W=30** | **14.31** | **300.1** |
| Zheng et al. (2017) | LSTM | 16.14 | 338 |
| Li, Ding & Sun (2018) | DCNN + window | see their Table 4 | see their Table 4 |
| Alomari et al. | RF/XGBoost/NGBoost ensemble | 11.8 | — |
| Ahmed (2025) | Attention-LSTM + XGBoost | see paper (FD001) | — |
| IMSA (2025) | LSTM vs. XGBoost vs. Chronos | LSTM 14.32 (best of three) | — |

Our XGBoost beats the Zheng et al. (2017) LSTM baseline on both metrics and
lands close to the IMSA (2025) LSTM number, but is still behind the Alomari
et al. tree ensemble — a reasonable place for a first window-statistics +
XGBoost pass to land, and consistent with the "low-teens to mid-20s" sanity
band rather than an outlier in either direction.

### What the two metrics mean

- **RMSE** (root-mean-squared error, in cycles) — the plain, symmetric
  error. An RMSE of 13.6 means predictions are typically off by roughly two
  weeks' worth of flight cycles, in either direction.
- **NASA PHM-2008 score** — the scoring function from the original 2008
  PHM Society challenge this dataset comes from. It's **asymmetric on
  purpose**: predicting *more* remaining life than an engine actually has
  (a "late" call) is penalized far more harshly than predicting less (an
  "early" call), because in real operations a late call means skipping
  maintenance the engine actually needed. Formula, where `d = predicted − true`:
  - `d < 0` (early): `score = exp(−d/13) − 1`
  - `d ≥ 0` (late): `score = exp(d/10) − 1` — smaller denominator, grows faster
  It's unbounded and only meaningful as a relative comparison between models
  on the same data, which is exactly how it's used here.

## 5. Reproducing this

```bash
# from the project root
pip install -r requirements.txt

# retrain everything from scratch (~3 minutes)
cd src
py train.py

# run the demo app
cd ../app
py -m streamlit run app.py
```

`train.py` will overwrite `models/xgb_model.joblib`, `models/scaler.joblib`,
`models/feature_names.json`, and `models/metrics.json` with a fresh run.
Results should match Section 4 almost exactly — the only randomness is in
Random Forest and XGBoost, both seeded (`RANDOM_STATE = 42` in `config.py`).

## 6. The app

`app/app.py` is a Streamlit demo with two modes:

- **Bundled sample engines** — 5 test engines pre-loaded (`app/sample_engines/`),
  spanning true RUL from 20 cycles (near failure) to 145 (healthy), so the
  app is fully demoable with zero setup or upload.
- **Upload your own CSV** — accepts either a raw headerless C-MAPSS file or
  a CSV with the standard column headers.

The standout feature is the **history slider**: it lets you "rewind" an
engine and watch the predicted RUL update as more cycles become visible.
On engine 100 (true RUL = 20), the predicted RUL drops from 123 cycles
(at just 30 cycles of history) down to 28.5 cycles (at the full 198 cycles) —
which is the model correctly picking up the degradation trend as the engine
approaches failure, and a genuinely useful thing to show in a demo.

## 7. Honesty / scope notes

Straight from the operating brief this project ships under:

- Phase 01 only: CAD/DFM services and ground-side software prototypes.
  **No airworthiness claims, anywhere, ever.**
- Not trained on official C-MAPSS test labels — `RUL_FD001.txt` is used
  only for scoring, never for fitting any model.
- FD001 only for now (single condition, single fault mode). FD003 (adds a
  second fault mode), then multi-condition FD002/FD004, are the natural
  next steps — not a rewrite into a different architecture.
- No LSTM/sequence model yet — the window-statistics approach was
  intentionally chosen first because it's simpler to validate and already
  lands in the expected error range; a sequence model is a stretch goal,
  not a requirement to ship.

## 8. Citation

**Dataset:**
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation.* International
Conference on Prognostics and Health Management (PHM08), Denver, CO.
Data: NASA Prognostics Center of Excellence Data Repository.

**Metrics** (RMSE + NASA PHM-2008 score, implemented in `metrics.py`):
Saxena, A., Celaya, J., Balaban, E., Goebel, K., et al. (2008). *Metrics for
Evaluating Performance of Prognostic Techniques.* PHM08.
doi.org/10.1109/PHM.2008.4711436

**Fair comparison** (why a FD001 score isn't automatically comparable to a
PHM'08-challenge score — read before adding rows to the table in §4.1):
Ramasso, E., & Saxena, A. (2014). *Performance Benchmarking and Analysis of
Prognostic Methods for CMAPSS Datasets.* International Journal of Prognostics
and Health Management, 5(2). doi.org/10.36001/ijphm.2014.v5i2.2236

**Baselines compared against in §4.1:**
- Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long Short-Term
  Memory Network for Remaining Useful Life Estimation.* IEEE ICPHM.
  doi.org/10.1109/ICPHM.2017.7998311
- Li, X., Ding, Q., & Sun, J.-Q. (2018). *Remaining Useful Life Estimation in
  Prognostics Using Deep Convolution Neural Networks.* Reliability Engineering
  & System Safety, 172, 1–11. doi.org/10.1016/j.ress.2017.11.021
- Alomari, Y., et al. *RF + XGBoost + NGBoost Ensemble for RUL Estimation on
  C-MAPSS.*
- Ahmed, S. (2025). *Attention-LSTM + XGBoost Hybrid for RUL Estimation.*
  MATEC Web of Conferences.
- IMSA (2025). *From LSTM to Chronos: A Comparison for RUL Estimation on
  FD001.* ieeexplore.ieee.org/document/11167842
