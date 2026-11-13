# Urban Bike Sharing Demand Forecasting

Spatio-temporal demand forecasting and dispatch support built on a conditional baseline plus residual learning, with GBT, Transformer, LSTM and ConvLSTM models and a Streamlit dashboard.

## Structure

```
.gitignore
app/app.py
app/init__.py
app/static/leaflet.css
app/static/leaflet.js
config.py
data/datasets.json
data/processed/weather/weather_start.txt
dev/archives_20260816/apptest_event.py
dev/archives_20260816/apptest_event2.py
dev/archives_20260816/apptest_event_center.py
dev/archives_20260816/apptest_final_v4.py
dev/archives_old_chain/app/utils.py
dev/archives_old_chain/src/evaluate.py
dev/archives_old_chain/src/forecast.py
dev/archives_old_chain/src/period_model.py
dev/archives_old_chain/src/preprocess.py
dev/archives_old_chain/src/train_convlstm.py
dev/archives_old_chain/src/train_lstm.py
dev/archives_old_chain/src/train_xgboost.py
dev/archives_old_chain/src/weather_enhance.py
dev/tests/apptest_rolling.py
dev/tests/apptest_ui.py
dev/tests/selftest.py
requirements.txt
src/baseline.py
src/data_loader.py
src/dataset_registry.py
src/date_features.py
src/init__.py
src/metrics.py
src/period_trainer.py
src/utils.py
src/weather.py
启动系统.bat
```

## Setup

Install dependencies from `requirements.txt` when present:

```bash
pip install -r requirements.txt
```

Data files, model weights and generated outputs are not included; place them under the paths referenced by the configuration before running.
