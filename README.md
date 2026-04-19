# itcs5154project

# Ocean Wave & SST Deep Learning Models

This project builds deep learning models to predict significant wave height (SWH)
and sea surface temperature (SST) using data from a single NDBC buoy (41110) off
the NC coast, inspired by three research papers on data-driven ocean modeling.

---

## Data

**Source:** `nc_coast_final.csv` — a single buoy (ID: 41110) dataset from 2020 onwards
pulled from Google Earth Engine exports.

**Columns:** `buoy_id`, `timestamp`, `wave_height`, `SST`, `temperature_2m`,
`total_precipitation`, `u_wind`, `v_wind`

**Preprocessing (`data_preprocessor.ipynb`):**
- Sorts by timestamp and cleans raw buoy records
- Splits chronologically: **70% train / 15% val / 15% test**

---

## Models

### Model 1 — Wave Height LSTM
Predicts next-day wave height using a sliding window of meteorological features.

- **Target:** `wave_height`
- **Features:** `u_wind`, `v_wind`, `temperature_2m`, `total_precipitation`, `SST`
- **Architecture:** LSTM (hidden=128, layers=2, dropout=0.3) → FC layers
- **Results:** RMSE = 0.2811 m, MAE = 0.1997 m, R² = 0.295

### Model 2 — Wind LSTM (SWH from Wind Speed)
Predicts wave height using derived wind speed and atmospheric features.

- **Target:** `wave_height`
- **Features:** `wind_speed` (derived from u/v), `temperature_2m`,
  `total_precipitation`, `wave_height_lag1/2/3`
- **Window:** 6 timesteps, ~1,361 samples
- **Architecture:** LSTM (hidden=64, layers=1, dropout=0.7) → FC layers

### Model 3 — Seat Surface Temperature LSTM (Multi-Lead SST Forecasting)
Predicts SST at multiple forecast horizons using lagged SST and weather features.

- **Target:** `SST`
- **Features:** `temperature_2m`, `u_wind`, `v_wind`, `total_precipitation`,
  `SST_lag1`, `SST_lag3`, `SST_lag7`, `SST_lag14`
- **Lead times:** 7, 30, and 90 days
- **Window:** 30 timesteps
- **Architecture:** LSTM (hidden=128, layers=2, dropout=0.3) → FC layers

### Model 4 — Significant Wave Height LSTM (SWH with Lagged Wave Features)
Predicts wave height using wind, atmospheric, and lagged wave height features.

- **Target:** `wave_height`
- **Features:** `u_wind`, `v_wind`, `temperature_2m`, `total_precipitation`,
  `wave_height_lag1/2/3`
- **Window:** 30 timesteps
- **Architecture:** LSTM (hidden=64, layers=2, dropout=0.3) → FC layers
- **Results:** Baseline RMSE = 0.2864 m

---

## Reference Papers

| File | Description |
|------|-------------|
| `paper1.pdf` | KIST-Ocean — U-shaped visual attention network for global 3D ocean simulation |
| `paper2.pdf` | AUWave — MLP + U-Net with self-attention for sparse wave field reconstruction |
| `paper3.pdf` | GWSM4C — CNN-based global wave surrogate model for climate simulation |

---

## Stack

Python · PyTorch · scikit-learn · Google Earth Engine · Google Colab