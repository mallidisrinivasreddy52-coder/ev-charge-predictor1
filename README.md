# EV Charge Time Predictor

PT1 demo: predicts EV charging time with **Linear Regression** and a **Decision Tree** (scikit-learn) and compares them in a small web app. There are two versions of the same app:

- **Streamlit** (`streamlit_app.py`): the shortest code. Start it with `streamlit run streamlit_app.py`.
- **Flask** (`app.py`): a classic website with a Python backend. Start it with `python app.py`.

Both use the same `ml.py` and `simulator.py`. **New to programming? Read `START_HERE.md` first.**

## Run it in VS Code

1. Install **Python 3.10 or newer** from python.org (tick "Add Python to PATH" on Windows).
2. In VS Code choose **File > Open Folder** and open this `ev-charge-predictor` folder. Install the **Python** extension if VS Code asks.
3. Open a terminal in VS Code (**Terminal > New Terminal**) and create a virtual environment:

   Windows:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   macOS / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
5. Start the app (Flask version; for Streamlit use `streamlit run streamlit_app.py`):
   ```
   python app.py
   ```
   You can also press **F5** and pick "Run EV Charge Time Predictor".
6. Open **http://127.0.0.1:5000** in your browser. Press `Ctrl+C` in the terminal to stop.

The first start creates `data/ev_sessions.csv`. To make a new copy of the dataset, run `python generate_data.py`.

## What is in the folder

| File | Purpose |
|---|---|
| `streamlit_app.py` | The Streamlit version of the app |
| `app.py` | Flask server and the two API routes (`/api/summary`, `/api/predict`) |
| `ml.py` | Loads the data, trains both models, scores them, makes predictions |
| `simulator.py` | The charging rules that create the simulated dataset |
| `generate_data.py` | Writes `data/ev_sessions.csv` |
| `templates/index.html`, `static/` | The web page (HTML, CSS, JavaScript) |

## Use your own PT1 dataset

The included data is **simulated**, not measured from real vehicles. To use your real dataset, make a CSV with these columns:

`battery_kwh, start_soc, target_soc, charger_kw, temp_c, charging_time_min`

Then start the app with the file path set:

Windows (Command Prompt):
```
set EV_DATA_CSV=C:\path\to\my_data.csv
python app.py
```
PowerShell:
```
$env:EV_DATA_CSV="C:\path\to\my_data.csv"
python app.py
```
macOS / Linux:
```
EV_DATA_CSV=/path/to/my_data.csv python app.py
```

With your own file the simulator reference and the "your session" markers are hidden, because they only make sense for simulated data. The models, scores and chart all use your rows.

## How the models are set up

- 80% of the rows train the models and 20% are held back for scoring (`random_state=42`).
- Inputs: battery capacity, starting charge, target charge, charger power, temperature, and energy to add (battery capacity x charge range).
- Decision Tree settings (maximum depth, minimum sessions per leaf) can be changed with the sliders on the page.
