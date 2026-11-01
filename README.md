# Engine Predictive Maintenance

![Pipeline](https://github.com/nirmalabtech/PredictiveMaintenance/actions/workflows/pipeline.yml/badge.svg)

Classifies engine condition (needs maintenance vs normal) from six sensor readings.

| Folder | Contents |
|---|---|
| `predictive_maintenance/data/` | Registered dataset (`engine_data.csv`) |
| `predictive_maintenance/model_building/` | Pipeline scripts: `data_register.py`, `prep.py`, `train.py` |
| `predictive_maintenance/deployment/` | Streamlit app, its requirements and the model committed by the pipeline |
| `predictive_maintenance/reports/` | Model comparison and tuning results written by the pipeline |
| `.github/workflows/pipeline.yml` | GitHub Actions workflow: register -> prep -> train -> deploy |

Live app: https://predictivemaintenance-kyfmh7zx4mgtwnvgnxwc2f.streamlit.app/
