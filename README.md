# EnergySmart - Electricity Consumption Forecasting Using Machine Learning

EnergySmart is an intelligent energy management system that predicts future electricity consumption from historical usage, calendar features, weather conditions, and seasonal patterns.

The project includes a Streamlit dashboard for uploading data, viewing trends, training a machine learning model, comparing actual and predicted usage, and forecasting future demand.

## Features

- Electricity usage forecasting with a Random Forest regression model
- Automatic date, time, day, month, weekday, and season feature extraction
- Weather-aware prediction using temperature, humidity, and wind speed
- Interactive charts for actual usage, predicted usage, and forecasted demand
- Built-in sample dataset for quick testing
- CSV upload support for custom electricity consumption data
- Login and registration system with a local SQLite database
- Save electricity readings into SQLite and view stored records
- Model performance metrics including MAE, RMSE, and R2 score

## Project Structure

```text
EnergySmart/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- run_app.bat
|-- install_to_desktop.bat
|-- install_to_desktop.ps1
|-- launch_energysmart.vbs
|-- assets/
|   `-- energysmart.ico
|-- data/
|   |-- sample_energy_data.csv
|   `-- energysmart.db
`-- src/
    |-- __init__.py
    |-- auth.py
    |-- features.py
    `-- model.py
```

`energysmart.db` is created automatically when the app starts.

Import-ready database files are also included:

```text
database/energysmart_import.db
database/energysmart_import.sql
```

Open `energysmart_import.db` directly in DB Browser for SQLite, or import `energysmart_import.sql` into a new SQLite database.

## SQLite Database

The app uses a local SQLite database at:

```text
data/energysmart.db
```

It contains two tables:

```text
users
energy_readings
```

After logging in, use **Save Current Data** in the sidebar to store the current sample or uploaded electricity data in SQLite. Open the **SQLite Database** tab to view saved electricity records and registered users.

You can also add one electricity reading manually from the **SQLite Database** tab using the **Save Reading To SQLite** form.

## Dataset Format

Your CSV file should contain these columns:

```text
timestamp,temperature,humidity,wind_speed,consumption_kwh
```

Example:

```text
2026-01-01 00:00:00,24.5,62,8.1,3.42
```

## How To Run

1. Install Python 3.10 or later.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
python -m streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal.
5. Create an account on the login screen, then sign in.

## Open It Directly From Your Desktop (Windows)

You don't need to open a terminal every time. Set this up once:

1. Keep `install_to_desktop.bat`, `install_to_desktop.ps1`,
   `launch_energysmart.vbs`, and `energysmart.ico` together in this project
   folder. The icon can sit either directly in this folder or inside an
   `assets` subfolder — the script checks both.
2. Double-click `install_to_desktop.bat`.
3. An **EnergySmart** icon — with its own lightning-bolt logo — appears on
   your Desktop.

From then on, just double-click that Desktop icon. It launches the app with
**no terminal window** — the dashboard just opens straight in your browser. Behind
the scenes it still runs `run_app.bat`, and the very first time (while packages are
being installed) it briefly shows the terminal so you can see that progress; every
launch after that is silent. If Python isn't installed, you'll get a plain pop-up
message telling you so, instead of nothing happening.

If you'd rather always see the terminal (for troubleshooting, say), you can still
double-click `run_app.bat` directly from this folder any time.

**If Windows shows a blue "Windows protected your PC" screen:** this happens on some
managed/lab PCs for downloaded scripts. Click **More info**, then **Run anyway**.

**If you still see an error creating the icon:** it will tell you exactly what
went wrong, and you can always fall back to double-clicking `run_app.bat` directly.

**If the icon still looks like the default one:** this is almost always one of
two things:

- `energysmart.ico` isn't in this folder or its `assets` subfolder — the script
  now prints both exact paths it checked if it can't find the file.
- Windows' icon cache is showing a stale picture. The script clears that cache
  and briefly restarts Explorer on its own after creating the icon. If it still
  looks unchanged afterward, sign out and back in once (or restart the PC) and it
  will pick up the new icon.

## Model

The application uses `RandomForestRegressor` from scikit-learn. It learns from:

- Previous electricity usage patterns
- Hour of day
- Day of week
- Month
- Weekend indicator
- Temperature
- Humidity
- Wind speed

## Goal

EnergySmart helps households, businesses, and institutions estimate electricity demand, reduce unnecessary usage, improve planning, and support sustainable energy management.
