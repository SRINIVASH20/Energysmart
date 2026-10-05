from __future__ import annotations

from datetime import datetime
from html import escape
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from src.auth import create_user, init_db, verify_user
from src.features import FEATURE_COLUMNS, add_time_features, build_future_frame, validate_dataset
from src.model import train_model


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA = BASE_DIR / "data" / "sample_energy_data.csv"
DB_PATH = BASE_DIR / "data" / "energysmart.db"
MAX_DATASET_ROWS = 20_000  # hard cap on readings per dataset / per account
SAMPLE_BILL_LIMIT = 20_000  # every sample dataset must stay below this two-month bill (Rs.)
REQUIRED_DATA_COLUMNS = [
    "timestamp", "temperature", "humidity", "wind_speed", "consumption_kwh"
]

st.set_page_config(page_title="EnergySmart", page_icon=":material/bolt:", layout="wide")

st.session_state.setdefault("theme", "Light")

TRANSLATIONS: dict[str, dict[str, str]] = {
    "English": {
        # Login page
        "app_title": "EnergySmart",
        "app_subtitle": "Forecast electricity demand with machine learning, weather signals, "
        "and your own usage history — all in one dashboard.",
        "feature_forecast_title": "Smart forecasting",
        "feature_forecast_body": "Random Forest model trained on your data",
        "feature_weather_title": "Weather-aware",
        "feature_weather_body": "Temperature, humidity & wind factored in",
        "feature_private_title": "Private & local",
        "feature_private_body": "Your readings stay in your own SQLite database",
        "login_tab": "Login", "register_tab": "Create account",
        "welcome_back": "Welcome back", "welcome_back_sub": "Sign in to pick up right where you left off.",
        "create_account_heading": "Create your account", "create_account_sub": "It takes less than a minute to get started.",
        "email": "Email", "password": "Password", "name": "Name", "confirm_password": "Confirm password",
        "login_button": "Login", "create_account_button": "Create account",
        # Sidebar
        "language": "Language", "signed_in_as": "Signed in as", "logout": "Logout",
        "theme_light": "Switch to light theme", "theme_dark": "Switch to dark theme",
        "data_source": "Data source", "location_type": "Location type", "forecast_settings": "Forecast settings",
        "forecast_hours": "Forecast hours", "showing": "Showing",
        "sample_data": "Sample data", "sqlite_readings": "SQLite readings", "upload_csv": "Upload CSV",
        "sample_data_suffix": "sample data", "your_saved_readings": "your saved SQLite readings",
        "your_uploaded_csv": "your uploaded CSV",
        "village": "Village", "town": "Town", "city": "City",
        # Workspace tabs
        "dashboard": "Dashboard", "forecast": "Forecast", "data": "Data", "sqlite_database": "SQLite database",
        "workspace": "Workspace",
        "dashboard_subtitle": "Live trends, hourly patterns, and how your latest reading compares with average. Showing {label}.",
        "forecast_subtitle": "Model-driven demand predictions, accuracy scores, and an estimated TN EB bill. Trained on {label}.",
        "data_subtitle": "Currently showing {label} — {count} readings.",
        "sqlite_subtitle": "Every reading you've saved, plus a form to add one manually.",
        # Metrics
        "records": "Records", "average_usage": "Average usage", "latest_usage": "Latest usage",
        "dataset_preview": "Dataset preview",
        "model_fit": "Model fit (R²)", "mae": "Mean Absolute Error", "rmse": "Root Mean Squared Error", "r2_score": "R² Score",
        "usage_breakdown": "Usage breakdown", "usage_level": "Usage level",
        "usage_breakdown_caption": "How readings in this date range split into low, medium, and high demand.",
        "pct_of_readings": "{pct:.0f}% of readings",
        "dlg_welcome_title": "Welcome to EnergySmart",
        "guest_name": "there",
        "dlg_hi": "Hi {name}, glad to have you here.",
        "dlg_map_intro": "Here's a quick map of what each section does:",
        "dlg_map_dashboard": "- **Dashboard** — trends, hourly averages, and how your latest reading compares with average.",
        "dlg_map_forecast": "- **Forecast** — trains the model and previews upcoming demand plus an estimated TN EB bill.",
        "dlg_map_data": "- **Data** — the dataset currently loaded (sample, upload, or saved readings).",
        "dlg_map_sqlite": "- **SQLite database** — save readings, add one manually, and download saved records.",
        "dlg_welcome_caption": "Use the sidebar to switch data sources, tune forecast settings, or log out. Look for the info icons for more details.",
        "got_it": "Got it",
        "close": "Close",
        "reading_guide_btn": "Reading guide",
        "view_tariff_btn": "View tariff slabs used",
        "dlg_guide_title": "Reading guide",
        "dlg_guide_consumption": "**Consumption (kWh):** The electricity used during one recorded hour.",
        "dlg_guide_average": "**Average usage:** The typical hourly usage across the selected date range.",
        "dlg_guide_weather": "**Temperature, humidity, and wind speed:** Weather signals the forecast model uses to estimate demand.",
        "dlg_guide_status": "**Status badge:** Compares the latest reading with the average for your selected date range.",
        "dlg_fs_title": "About forecast settings",
        "dlg_fs_hours": "- **Forecast hours:** How many hours ahead the model predicts, starting from your most recent reading.",
        "dlg_fs_weather": "- **Expected temperature / humidity / wind speed:** Weather assumptions used for every forecasted hour. They default to your recent 24-hour average, and you can adjust them to test different conditions.",
        "dlg_fs_caption": "Changing these values updates the Forecast workspace the next time you view it.",
        "dlg_tariff_title": "Tamil Nadu EB domestic rates used in this estimate",
        "tariff_total_le_500": "Total use up to 500 units",
        "tariff_total_gt_500": "Total use above 500 units",
        "tariff_col_total": "Two-month bill total",
        "tariff_col_slab": "Units in the slab",
        "tariff_col_charge": "Energy charge",
        "slab_first": "First {n} units",
        "slab_range": "{a}-{b} units",
        "slab_above": "Above {n} units",
        "rs_per_unit": "Rs. {p} per unit",
        "free": "Free",
        "dlg_tariff_caption": "TN EB domestic LT-IA energy-charge estimate. Fixed charges, taxes, arrears, meter-related charges, and future tariff changes are not included. Check the official TNPDCL bill calculator before payment.",
        "dlg_sqlite_title": "About the SQLite database",
        "dlg_sqlite_intro": "Your readings are stored locally in a SQLite database at `data/energysmart.db`.",
        "dlg_sqlite_users": "- **users** — your account details (name, email, and a hashed password).",
        "dlg_sqlite_readings": "- **energy_readings** — every reading you've saved or added, linked to your account.",
        "dlg_sqlite_caption": "Only your own readings are shown here. Use the form above to add one manually, or save the currently loaded dataset from the sidebar.",
        "warn_no_saved": "No saved readings yet. Showing sample data instead.",
        "err_invalid_login": "Invalid email or password.",
        "err_pw_mismatch": "Passwords do not match.",
        "live_none": "Live status will appear after you save or add a reading.",
        "live_latest": "Latest saved reading: {ts}",
        "live_consumption": "Live consumption",
        "upload_csv_label": "Upload electricity CSV",
        "location_help": "Each location has its own realistic demand profile — villages run low and flat, towns show clear morning/evening peaks, cities add heavier cooling and commercial load.",
        "fs_info_help": "What do these settings control?",
        "exp_temp": "Expected temperature",
        "exp_humidity": "Expected humidity",
        "exp_wind": "Expected wind speed",
        "live_readings": "Live readings",
        "gen_live": "Generate live reading",
        "save_current": "Save current data",
        "toast_live_saved": "A new live reading was saved to SQLite.",
        "toast_saved_n": "Saved {n} readings to SQLite.",
        "refresh_live": "Refresh live status",
        "page_tagline": "Dynamic electricity consumption forecasting using machine learning",
        "badge_live": "Live energy insights",
        "badge_ready": "Forecast ready",
        "r2_help": "R² (R-squared): how much of the variation in electricity usage the model explains. 1.00 is a perfect fit; 0.00 means it's no better than guessing the average.",
        "mae_help": "MAE (Mean Absolute Error): on average, how far off each prediction is from the actual reading, in kWh. Lower is better.",
        "rmse_help": "RMSE (Root Mean Squared Error): similar to MAE but penalizes large misses more heavily, in kWh. Lower is better.",
        "date_range": "Date range",
        "status_higher": "Higher than usual",
        "status_lower": "Lower than usual",
        "status_close": "Close to usual",
        "msg_higher": "This reading is above the selected-period average. Consider shifting flexible appliance use away from this time.",
        "msg_lower": "This reading is below the selected-period average. It is a lower-demand part of the day.",
        "msg_close": "This reading is close to the selected-period average, so your demand is steady.",
        "latest_reading_explained": "Latest reading explained",
        "electricity_used": "Electricity used",
        "from_average": "{d:+.2f} kWh from average",
        "temperature": "Temperature",
        "humidity_line": "Humidity: {h:.0f}%",
        "usage_trend": "Electricity usage trend",
        "avg_by_hour": "Average usage by hour",
        "temp_and_consumption": "Temperature and consumption",
        "live_sqlite_status": "Live SQLite status",
        "forecast_at_glance": "Forecast at a glance",
        "exp_units_label": "Expected electricity use for two months (units)",
        "exp_units_help": "This starts with a 60-day projection based on the forecast. You can change it to match your home usage.",
        "expected_energy": "Expected energy",
        "tn_eb_estimate": "TN EB two-month estimate",
        "peak_demand": "Peak demand",
        "compared_usual": "Compared with usual",
        "free_scheme_within": "Tamil Nadu domestic free-electricity scheme: {n:.0f} units are free for this two-month estimate because usage is within 500 units.",
        "free_scheme_above": "Tamil Nadu domestic free-electricity scheme: this estimate is above 500 units, so it includes the existing {n:.0f} free units for the two-month bill.",
        "demand_high": "Demand is expected to be higher than usual. Reduce non-essential usage around {time} to avoid the forecast peak.",
        "demand_low": "Demand is expected to be lower than usual. This is a good period for energy-intensive tasks.",
        "demand_normal": "Demand is expected to stay close to normal. The lowest forecast is {v:.2f} kWh at {time}.",
        "actual_vs_pred": "Actual and predicted consumption",
        "future_forecast": "Future demand forecast",
        "hourly_schedule": "Hourly forecast schedule",
        "col_time": "Time",
        "col_expected": "Expected usage",
        "col_timestamp": "Timestamp",
        "col_temperature": "Temperature",
        "col_humidity": "Humidity",
        "col_wind": "Wind speed",
        "col_consumption": "Consumption",
        "col_saved": "Saved",
        "normal": "Normal",
        "download_forecast": "Download forecast CSV",
        "download_saved": "Download saved records CSV",
        "sqlite_info_help": "What is stored here?",
        "f_date": "Date",
        "f_time": "Time",
        "f_consumption": "Consumption (kWh)",
        "save_reading": "Save reading to SQLite",
        "reading_saved": "Reading saved in the SQLite database.",
        "no_readings": "No readings saved yet. Use the live reading button or save current data.",
        "units_title": "Units & estimated bill",
        "units_note": "1 unit = 1 kWh of electricity. Bills are estimated with the TN EB domestic slabs for a 60-day (two-month) period.",
        "units_two_month": "{u:,.0f} units / 2 months",
        "units_per_day": "{u:.1f} units per day",
        "units_bill": "Estimated bill: Rs. {b:,.0f}",
        "units_below": "Below Rs. {limit:,}",
        "units_above": "Above Rs. {limit:,}",
        "units_selected": "Selected",
        "units_dataset_total": "Dataset total: {u:,.0f} units over {d:.0f} days",
        "units_sidebar": "≈ {u:,.0f} units / 2 months · Rs. {b:,.0f}",
        "units_help": "Average electricity used per reading. 1 unit = 1 kWh.",
        "dlg_reg_title": "Account created!",
        "dlg_reg_welcome": "Welcome, {name}!",
        "dlg_reg_body": "Your account for {email} is ready.",
        "dlg_reg_next": "Next step: open the **{login}** tab and sign in with your email and password.",
        "dlg_reg_privacy": "Your readings stay private in your own SQLite database. Each account keeps up to {max:,} readings.",
        "reg_success_line": "Account created. You can log in now.",
        "warn_capped": "Dataset limited to the latest {max:,} readings — {dropped:,} older readings were left out.",
        "toast_capped": "Only the latest {max:,} readings are used.",
        "upload_limit_hint": "Datasets are limited to {max:,} readings. If your file has more, only the newest {max:,} are used.",
        "dlg_sqlite_limit": "- **Limit:** each account keeps at most {max:,} readings; the oldest are removed automatically when the limit is reached.",
        "bd_range": "Range",
        "bd_avg": "Average",
        "bd_energy_share": "Share of energy",
        "bd_peak_time": "Most common time",
        "bd_avg_temp": "Avg temperature",
        "bd_bar_readings": "Share of readings",
        "bd_bar_energy": "Share of energy (kWh)",
        "period_night": "Night (12–6 AM)",
        "period_morning": "Morning (6 AM–12 PM)",
        "period_afternoon": "Afternoon (12–6 PM)",
        "period_evening": "Evening (6 PM–12 AM)",
        "period_short_night": "night",
        "period_short_morning": "morning",
        "period_short_afternoon": "afternoon",
        "period_short_evening": "evening",
        "bd_insight_energy": "‘{level}’ readings are {rp:.0f}% of all readings but use {ep:.0f}% of the energy.",
        "bd_insight_when": "Most ‘{level}’ readings ({p:.0f}%) happen in the {period}.",
        "bd_insight_weather": "‘{hi_label}’ readings average {hot:.1f} °C versus {cool:.1f} °C for ‘{lo_label}’ readings.",
        "bd_tab_hour": "By hour",
        "bd_tab_period": "Time of day",
        "bd_tab_day": "Weekday vs weekend",
        "bd_tab_weather": "Weather",
        "bd_hour_caption": "How many readings fall into each level at every hour of the day.",
        "bd_period_caption": "Share of each part of the day's readings that are Low, Medium or High.",
        "bd_day_caption": "Share of weekday and weekend readings at each level.",
        "bd_weather_caption": "Average usage and weather for the readings at each level.",
        "col_period": "Period",
        "col_daytype": "Day type",
        "col_level": "Level",
        "col_avg_kwh": "Avg usage (kWh)",
        "col_readings": "Readings",
        "col_avg_temp": "Avg temperature (°C)",
        "col_avg_hum": "Avg humidity (%)",
        "col_avg_wind": "Avg wind (km/h)",
        "bd_weekday": "Weekday",
        "bd_weekend": "Weekend",
        "tips_title": "Suggestions to save energy",
        "tips_caption": "A few practical ideas based on the patterns in this data.",
        "tip_peak_shift_title": "Shift flexible loads off peak hours",
        "tip_peak_shift_detail": "Usage is highest around {peak}. Running washing machines, water heaters, or EV charging around {off} instead can lower your peak demand and bill.",
        "tip_cooling_title": "Ease up on cooling during hot spells",
        "tip_cooling_detail": "Usage rises noticeably as it gets hotter in this data. Setting the AC a couple of degrees warmer, using fans alongside it, and closing curtains in direct sun can meaningfully cut cooling costs.",
        "tip_weekend_high_title": "Weekend usage runs higher",
        "tip_weekend_high_detail": "Readings are higher on weekends than weekdays. Spreading out heavy chores (laundry, cooking-heavy meals) across the week can smooth out your demand.",
        "tip_weekday_high_title": "Weekday usage runs higher",
        "tip_weekday_high_detail": "Readings are higher on weekdays than weekends. If that's from appliances left running during work hours, a timer or smart plug can help cut it down.",
        "tip_high_share_title": "A large share of readings are High",
        "tip_high_share_detail": "{pct:.0f}% of readings fall in the High band. Checking for an always-on heavy appliance (old fridge, water heater, AC) is often the quickest way to bring this down.",
        "tip_city_title": "City-scale cooling and commercial load",
        "tip_city_detail": "City usage tends to be driven by cooling and commercial activity. Scheduling AC with a timer and favoring energy-efficient (5-star) appliances tends to help most here.",
        "tip_village_title": "Keep the low, steady baseline",
        "tip_village_detail": "Village usage is already low and flat. The biggest wins are usually in lighting (LEDs) and avoiding standby power, since there isn't much heavy cooling load to begin with.",
        "tip_led_title": "Switch to LED lighting",
        "tip_led_detail": "LED bulbs use about 75-80% less electricity than incandescent bulbs for the same brightness, and last far longer.",
        "tip_standby_title": "Unplug idle electronics",
        "tip_standby_detail": "TVs, chargers, and set-top boxes left plugged in still draw small amounts of standby power around the clock. A power strip you switch off at night adds up over a month.",
        "room_usage_title": "Room-wise usage",
        "room_usage_caption": "A typical household split applied to your actual usage, as a rough guide to where it's going.",
        "room_usage_disclaimer": "Estimated from common usage patterns, not measured per room — this app doesn't have per-room sensors.",
        "room_est_cost": "≈ Rs. {v:,.0f}",
        "room_kitchen": "Kitchen",
        "room_living": "Living room & entertainment",
        "room_bedroom": "Bedroom & AC",
        "room_water_heater": "Water heater",
        "room_lighting_other": "Lighting & other",
        "reasons_title": "Why is usage {level}?",
        "reasons_forecast_title": "Why does demand peak around {time}?",
        "reasons_caption": "The causes behind this level, worked out from the readings in this dataset. The outlined card is the main cause.",
        "reasons_summary_above": "{v:.2f} kWh is {p:.0f}% above the average of {avg:.2f} kWh.",
        "reasons_summary_below": "{v:.2f} kWh is {p:.0f}% below the average of {avg:.2f} kWh.",
        "reasons_summary_near": "{v:.2f} kWh is close to the average of {avg:.2f} kWh.",
        "tag_raises": "Raises usage",
        "tag_lowers": "Lowers usage",
        "tag_neutral": "No big effect",
        "tag_context": "Context",
        "tag_main": "Main cause",
        "est_effect": " (about {e:+.2f} kWh in this data)",
        "reason_peak_hour_title": "Peak-demand hour",
        "reason_peak_hour_detail": "{hour} is one of the busiest hours in this data — usage here averages {v:.2f} kWh, {p:.0f}% above the daily average.",
        "reason_offpeak_hour_title": "Quiet hour",
        "reason_offpeak_hour_detail": "{hour} is one of the quietest hours in this data — usage here averages {v:.2f} kWh, {p:.0f}% below the daily average.",
        "reason_normal_hour_title": "Typical hour",
        "reason_normal_hour_detail": "{hour} is an ordinary hour; usage here averages {v:.2f} kWh.",
        "reason_hot_title": "Hot weather — heavier cooling",
        "reason_hot_detail": "{temp:.1f} °C is {d:.1f} °C above the usual {avg:.1f} °C, so fans and air conditioners run harder{est}.",
        "reason_cool_title": "Cooler weather — less cooling",
        "reason_cool_detail": "{temp:.1f} °C is {d:.1f} °C below the usual {avg:.1f} °C, so less cooling is needed{est}.",
        "reason_temp_normal_title": "Normal temperature",
        "reason_temp_normal_detail": "{temp:.1f} °C is close to the usual {avg:.1f} °C, so the weather isn't pushing usage either way.",
        "reason_weekend_title": "Weekend pattern",
        "reason_weekend_detail": "This is a weekend. Weekends average {w:.2f} kWh versus {d:.2f} kWh on weekdays in this data.",
        "reason_weekday_title": "Weekday pattern",
        "reason_weekday_detail": "This is a weekday. Weekdays average {d:.2f} kWh versus {w:.2f} kWh on weekends in this data.",
        "reason_humid_title": "High humidity",
        "reason_humid_detail": "{h:.0f}% humidity feels hotter, so fans and air conditioners tend to run longer.",
        "reason_windy_title": "Strong wind",
        "reason_windy_detail": "A {w:.1f} km/h wind gives natural ventilation, which reduces the need for cooling.",
        "reason_none_title": "Following the usual pattern",
        "reason_none_detail": "No single strong driver stands out; usage is following its normal routine.",
        "reason_loc_title": "Location profile: {loc}",
        "reason_loc_village": "Village homes run few heavy appliances, so the base load stays low and flat.",
        "reason_loc_town": "Towns show clear morning and evening peaks as households and small shops follow a daily routine.",
        "reason_loc_city": "Cities carry heavy cooling and commercial load, which lifts the base level and sharpens rush-hour peaks.",
        "low": "Low", "medium": "Medium", "high": "High", "saved_records": "Saved records",
    },
    "தமிழ்": {
        # Login page
        "app_title": "EnergySmart",
        "app_subtitle": "இயந்திரக் கற்றல், வானிலை அறிகுறிகள், உங்கள் சொந்த பயன்பாட்டு வரலாற்றைக் "
        "கொண்டு மின் தேவையை முன்னறிவிக்கவும் — அனைத்தும் ஒரே டாஷ்போர்டில்.",
        "feature_forecast_title": "திறமையான முன்னறிவிப்பு",
        "feature_forecast_body": "உங்கள் தரவில் பயிற்சி பெற்ற Random Forest மாதிரி",
        "feature_weather_title": "வானிலை அறிந்த",
        "feature_weather_body": "வெப்பநிலை, ஈரப்பதம் & காற்று வேகம் கணக்கில் எடுத்துக்கொள்ளப்படுகிறது",
        "feature_private_title": "தனிப்பட்ட & உள்ளூர்",
        "feature_private_body": "உங்கள் அளவீடுகள் உங்கள் சொந்த SQLite தரவுத்தளத்தில் பாதுகாப்பாக இருக்கும்",
        "login_tab": "உள்நுழைய", "register_tab": "கணக்கு உருவாக்க",
        "welcome_back": "மீண்டும் வருக", "welcome_back_sub": "நீங்கள் விட்ட இடத்திலிருந்தே தொடர உள்நுழையவும்.",
        "create_account_heading": "உங்கள் கணக்கை உருவாக்கவும்", "create_account_sub": "ஒரு நிமிடத்திற்கும் குறைவான நேரம் மட்டுமே ஆகும்.",
        "email": "மின்னஞ்சல்", "password": "கடவுச்சொல்", "name": "பெயர்", "confirm_password": "கடவுச்சொல்லை உறுதிப்படுத்தவும்",
        "login_button": "உள்நுழைய", "create_account_button": "கணக்கு உருவாக்க",
        # Sidebar
        "language": "மொழி", "signed_in_as": "உள்நுழைந்துள்ளவர்", "logout": "வெளியேறு",
        "theme_light": "வெளிர் தீம் மாற்று", "theme_dark": "இருண்ட தீம் மாற்று",
        "data_source": "தரவு மூலம்", "location_type": "இட வகை", "forecast_settings": "முன்னறிவிப்பு அமைப்புகள்",
        "forecast_hours": "முன்னறிவிப்பு மணிநேரங்கள்", "showing": "காட்டப்படுகிறது",
        "sample_data": "மாதிரி தரவு", "sqlite_readings": "சேமிக்கப்பட்ட அளவீடுகள்", "upload_csv": "CSV பதிவேற்று",
        "sample_data_suffix": "மாதிரி தரவு", "your_saved_readings": "உங்கள் சேமிக்கப்பட்ட அளவீடுகள்",
        "your_uploaded_csv": "நீங்கள் பதிவேற்றிய CSV",
        "village": "கிராமம்", "town": "நகரம்", "city": "பெருநகரம்",
        # Workspace tabs
        "dashboard": "டாஷ்போர்டு", "forecast": "முன்னறிவிப்பு", "data": "தரவு", "sqlite_database": "SQLite தரவுத்தளம்",
        "workspace": "பணியிடம்",
        "dashboard_subtitle": "நேரடி போக்குகள், மணிநேர முறைகள், உங்கள் சமீபத்திய அளவீடு சராசரியுடன் எவ்வாறு "
        "ஒப்பிடுகிறது என்பதைக் காட்டுகிறது. {label} காட்டப்படுகிறது.",
        "forecast_subtitle": "மாதிரி அடிப்படையிலான தேவை முன்னறிவிப்புகள், துல்லியம் மற்றும் மதிப்பிடப்பட்ட "
        "TN EB கட்டணம். {label} இல் பயிற்சி பெற்றது.",
        "data_subtitle": "தற்போது {label} காட்டப்படுகிறது — {count} அளவீடுகள்.",
        "sqlite_subtitle": "நீங்கள் சேமித்த ஒவ்வொரு அளவீடும், மேலும் ஒன்றை கைமுறையாக சேர்க்க ஒரு படிவம்.",
        # Metrics
        "records": "பதிவுகள்", "average_usage": "சராசரி பயன்பாடு", "latest_usage": "சமீபத்திய பயன்பாடு",
        "dataset_preview": "தரவுத்தொகுப்பு முன்னோட்டம்",
        "model_fit": "மாதிரி பொருத்தம் (R²)", "mae": "சராசரி முழுமையான பிழை", "rmse": "வர்க்க மூல சராசரி பிழை",
        "r2_score": "R² மதிப்பெண்", "usage_breakdown": "பயன்பாட்டு பிரிவு", "usage_level": "பயன்பாட்டு நிலை",
        "usage_breakdown_caption": "இந்த தேதி வரம்பில் உள்ள அளவீடுகள் குறைவு, நடுத்தரம், அதிகம் என பிரிக்கப்படும் விதம்.",
        "pct_of_readings": "{pct:.0f}% அளவீடுகள்",
        "dlg_welcome_title": "EnergySmart-க்கு வரவேற்கிறோம்",
        "guest_name": "நண்பரே",
        "dlg_hi": "வணக்கம் {name}, உங்களை இங்கு காண மகிழ்ச்சி.",
        "dlg_map_intro": "ஒவ்வொரு பிரிவும் என்ன செய்கிறது என்பதன் சுருக்கம்:",
        "dlg_map_dashboard": "- **டாஷ்போர்டு** — போக்குகள், மணிநேர சராசரிகள், உங்கள் சமீபத்திய அளவீடு சராசரியுடன் எவ்வாறு ஒப்பிடுகிறது.",
        "dlg_map_forecast": "- **முன்னறிவிப்பு** — மாதிரியைப் பயிற்றுவித்து, வரவிருக்கும் தேவையையும் மதிப்பிடப்பட்ட TN EB கட்டணத்தையும் காட்டுகிறது.",
        "dlg_map_data": "- **தரவு** — தற்போது ஏற்றப்பட்ட தரவுத்தொகுப்பு (மாதிரி, பதிவேற்றம் அல்லது சேமித்த அளவீடுகள்).",
        "dlg_map_sqlite": "- **SQLite தரவுத்தளம்** — அளவீடுகளைச் சேமிக்கவும், ஒன்றை கைமுறையாகச் சேர்க்கவும், சேமித்த பதிவுகளைப் பதிவிறக்கவும்.",
        "dlg_welcome_caption": "தரவு மூலங்களை மாற்ற, முன்னறிவிப்பு அமைப்புகளைச் சரிசெய்ய அல்லது வெளியேற பக்கப்பட்டியைப் பயன்படுத்தவும். கூடுதல் விவரங்களுக்கு தகவல் ஐகான்களைப் பாருங்கள்.",
        "got_it": "புரிந்தது",
        "close": "மூடு",
        "reading_guide_btn": "வாசிப்பு வழிகாட்டி",
        "view_tariff_btn": "பயன்படுத்திய கட்டண ஸ்லாப்களைக் காண்க",
        "dlg_guide_title": "வாசிப்பு வழிகாட்டி",
        "dlg_guide_consumption": "**நுகர்வு (kWh):** ஒரு பதிவு செய்யப்பட்ட மணிநேரத்தில் பயன்படுத்தப்பட்ட மின்சாரம்.",
        "dlg_guide_average": "**சராசரி பயன்பாடு:** தேர்ந்தெடுத்த தேதி வரம்பில் வழக்கமான மணிநேர பயன்பாடு.",
        "dlg_guide_weather": "**வெப்பநிலை, ஈரப்பதம், காற்று வேகம்:** தேவையை மதிப்பிட முன்னறிவிப்பு மாதிரி பயன்படுத்தும் வானிலை அறிகுறிகள்.",
        "dlg_guide_status": "**நிலை குறிச்சொல்:** தேர்ந்தெடுத்த தேதி வரம்பின் சராசரியுடன் சமீபத்திய அளவீட்டை ஒப்பிடுகிறது.",
        "dlg_fs_title": "முன்னறிவிப்பு அமைப்புகள் பற்றி",
        "dlg_fs_hours": "- **முன்னறிவிப்பு மணிநேரங்கள்:** உங்கள் சமீபத்திய அளவீட்டிலிருந்து தொடங்கி, மாதிரி எத்தனை மணிநேரம் முன்னதாகக் கணிக்கிறது.",
        "dlg_fs_weather": "- **எதிர்பார்க்கும் வெப்பநிலை / ஈரப்பதம் / காற்று வேகம்:** ஒவ்வொரு முன்னறிவிக்கப்பட்ட மணிநேரத்திற்கும் பயன்படுத்தப்படும் வானிலை அனுமானங்கள். இவை இயல்பாக உங்கள் சமீபத்திய 24 மணிநேர சராசரியாக இருக்கும்; வெவ்வேறு நிலைகளைச் சோதிக்க மாற்றலாம்.",
        "dlg_fs_caption": "இந்த மதிப்புகளை மாற்றினால், அடுத்த முறை முன்னறிவிப்புப் பகுதியைப் பார்க்கும்போது புதுப்பிக்கப்படும்.",
        "dlg_tariff_title": "இந்த மதிப்பீட்டில் பயன்படுத்தப்பட்ட தமிழ்நாடு EB வீட்டு மின் கட்டணங்கள்",
        "tariff_total_le_500": "மொத்த பயன்பாடு 500 அலகுகள் வரை",
        "tariff_total_gt_500": "மொத்த பயன்பாடு 500 அலகுகளுக்கு மேல்",
        "tariff_col_total": "இரு மாத பில் மொத்தம்",
        "tariff_col_slab": "ஸ்லாபில் உள்ள அலகுகள்",
        "tariff_col_charge": "மின் கட்டணம்",
        "slab_first": "முதல் {n} அலகுகள்",
        "slab_range": "{a}-{b} அலகுகள்",
        "slab_above": "{n} அலகுகளுக்கு மேல்",
        "rs_per_unit": "ரூ. {p} / அலகு",
        "free": "இலவசம்",
        "dlg_tariff_caption": "TN EB வீட்டு LT-IA மின் கட்டண மதிப்பீடு. நிலையான கட்டணங்கள், வரிகள், நிலுவைத் தொகை, மீட்டர் தொடர்பான கட்டணங்கள் மற்றும் எதிர்கால கட்டண மாற்றங்கள் சேர்க்கப்படவில்லை. செலுத்துவதற்கு முன் அதிகாரப்பூர்வ TNPDCL பில் கால்குலேட்டரைச் சரிபார்க்கவும்.",
        "dlg_sqlite_title": "SQLite தரவுத்தளம் பற்றி",
        "dlg_sqlite_intro": "உங்கள் அளவீடுகள் `data/energysmart.db` இல் உள்ள SQLite தரவுத்தளத்தில் உள்ளூரிலேயே சேமிக்கப்படுகின்றன.",
        "dlg_sqlite_users": "- **users** — உங்கள் கணக்கு விவரங்கள் (பெயர், மின்னஞ்சல், மறைகுறியாக்கப்பட்ட கடவுச்சொல்).",
        "dlg_sqlite_readings": "- **energy_readings** — நீங்கள் சேமித்த அல்லது சேர்த்த ஒவ்வொரு அளவீடும், உங்கள் கணக்குடன் இணைக்கப்பட்டது.",
        "dlg_sqlite_caption": "உங்கள் சொந்த அளவீடுகள் மட்டுமே இங்கே காட்டப்படும். மேலுள்ள படிவம் மூலம் ஒன்றை கைமுறையாகச் சேர்க்கலாம் அல்லது பக்கப்பட்டியிலிருந்து தற்போதைய தரவுத்தொகுப்பைச் சேமிக்கலாம்.",
        "warn_no_saved": "சேமித்த அளவீடுகள் இன்னும் இல்லை. அதற்குப் பதிலாக மாதிரி தரவு காட்டப்படுகிறது.",
        "err_invalid_login": "மின்னஞ்சல் அல்லது கடவுச்சொல் தவறானது.",
        "err_pw_mismatch": "கடவுச்சொற்கள் பொருந்தவில்லை.",
        "live_none": "அளவீட்டைச் சேமித்த அல்லது சேர்த்த பிறகு நேரடி நிலை தோன்றும்.",
        "live_latest": "சமீபத்தில் சேமித்த அளவீடு: {ts}",
        "live_consumption": "நேரடி நுகர்வு",
        "upload_csv_label": "மின் நுகர்வு CSV பதிவேற்றவும்",
        "location_help": "ஒவ்வொரு இடத்திற்கும் அதற்கே உரிய தேவை முறை உண்டு — கிராமங்களில் குறைவாகவும் சீராகவும், நகரங்களில் காலை/மாலை உச்சங்களுடனும், பெருநகரங்களில் கூடுதல் குளிரூட்டல் மற்றும் வணிகப் பயன்பாட்டுடனும் இருக்கும்.",
        "fs_info_help": "இந்த அமைப்புகள் எதைக் கட்டுப்படுத்துகின்றன?",
        "exp_temp": "எதிர்பார்க்கும் வெப்பநிலை",
        "exp_humidity": "எதிர்பார்க்கும் ஈரப்பதம்",
        "exp_wind": "எதிர்பார்க்கும் காற்று வேகம்",
        "live_readings": "நேரடி அளவீடுகள்",
        "gen_live": "நேரடி அளவீட்டை உருவாக்கு",
        "save_current": "தற்போதைய தரவைச் சேமி",
        "toast_live_saved": "புதிய நேரடி அளவீடு SQLite-இல் சேமிக்கப்பட்டது.",
        "toast_saved_n": "{n} அளவீடுகள் SQLite-இல் சேமிக்கப்பட்டன.",
        "refresh_live": "நேரடி நிலையைப் புதுப்பி",
        "page_tagline": "இயந்திரக் கற்றலைப் பயன்படுத்தி மாறும் மின் நுகர்வு முன்னறிவிப்பு",
        "badge_live": "நேரடி ஆற்றல் தகவல்கள்",
        "badge_ready": "முன்னறிவிப்பு தயார்",
        "r2_help": "R² (R-squared): மின் பயன்பாட்டின் மாறுபாட்டை மாதிரி எவ்வளவு விளக்குகிறது என்பதைக் காட்டுகிறது. 1.00 என்பது சரியான பொருத்தம்; 0.00 என்பது சராசரியை ஊகிப்பதை விட மேம்பட்டதல்ல.",
        "mae_help": "MAE (சராசரி முழுமையான பிழை): ஒவ்வொரு கணிப்பும் உண்மையான அளவீட்டிலிருந்து சராசரியாக எவ்வளவு விலகுகிறது, kWh-இல். குறைவாக இருப்பது நல்லது.",
        "rmse_help": "RMSE (வர்க்க மூல சராசரி பிழை): MAE போன்றதே, ஆனால் பெரிய தவறுகளுக்கு அதிக எடை தருகிறது, kWh-இல். குறைவாக இருப்பது நல்லது.",
        "date_range": "தேதி வரம்பு",
        "status_higher": "வழக்கத்தை விட அதிகம்",
        "status_lower": "வழக்கத்தை விடக் குறைவு",
        "status_close": "வழக்கத்தை ஒட்டியது",
        "msg_higher": "இந்த அளவீடு தேர்ந்தெடுத்த காலத்தின் சராசரியை விட அதிகம். மாற்றக்கூடிய சாதனப் பயன்பாட்டை இந்த நேரத்திலிருந்து விலக்கிப் பரிசீலிக்கவும்.",
        "msg_lower": "இந்த அளவீடு தேர்ந்தெடுத்த காலத்தின் சராசரியை விடக் குறைவு. இது நாளின் குறைந்த தேவை நேரம்.",
        "msg_close": "இந்த அளவீடு தேர்ந்தெடுத்த காலத்தின் சராசரிக்கு அருகில் உள்ளது, எனவே உங்கள் தேவை சீராக உள்ளது.",
        "latest_reading_explained": "சமீபத்திய அளவீட்டின் விளக்கம்",
        "electricity_used": "பயன்படுத்திய மின்சாரம்",
        "from_average": "சராசரியிலிருந்து {d:+.2f} kWh",
        "temperature": "வெப்பநிலை",
        "humidity_line": "ஈரப்பதம்: {h:.0f}%",
        "usage_trend": "மின் பயன்பாட்டுப் போக்கு",
        "avg_by_hour": "மணிநேரவாரி சராசரிப் பயன்பாடு",
        "temp_and_consumption": "வெப்பநிலையும் நுகர்வும்",
        "live_sqlite_status": "நேரடி SQLite நிலை",
        "forecast_at_glance": "ஒரே பார்வையில் முன்னறிவிப்பு",
        "exp_units_label": "இரண்டு மாதங்களுக்கு எதிர்பார்க்கும் மின் பயன்பாடு (அலகுகள்)",
        "exp_units_help": "இது முன்னறிவிப்பின் அடிப்படையில் 60 நாள் கணிப்புடன் தொடங்குகிறது. உங்கள் வீட்டுப் பயன்பாட்டுக்கு ஏற்ப மாற்றலாம்.",
        "expected_energy": "எதிர்பார்க்கும் ஆற்றல்",
        "tn_eb_estimate": "TN EB இரு மாத மதிப்பீடு",
        "peak_demand": "உச்ச தேவை",
        "compared_usual": "வழக்கத்துடன் ஒப்பிட",
        "free_scheme_within": "தமிழ்நாடு வீட்டு இலவச மின்சாரத் திட்டம்: பயன்பாடு 500 அலகுகளுக்குள் இருப்பதால் இந்த இரு மாத மதிப்பீட்டில் {n:.0f} அலகுகள் இலவசம்.",
        "free_scheme_above": "தமிழ்நாடு வீட்டு இலவச மின்சாரத் திட்டம்: இந்த மதிப்பீடு 500 அலகுகளுக்கு மேல் உள்ளதால், இரு மாத பில்லில் தற்போதுள்ள {n:.0f} இலவச அலகுகள் சேர்க்கப்பட்டுள்ளன.",
        "demand_high": "தேவை வழக்கத்தை விட அதிகமாக இருக்கும் என எதிர்பார்க்கப்படுகிறது. முன்னறிவிப்பு உச்சத்தைத் தவிர்க்க {time} அளவில் அத்தியாவசியமற்ற பயன்பாட்டைக் குறைக்கவும்.",
        "demand_low": "தேவை வழக்கத்தை விடக் குறைவாக இருக்கும் என எதிர்பார்க்கப்படுகிறது. அதிக மின் தேவைப்படும் பணிகளுக்கு இது நல்ல காலம்.",
        "demand_normal": "தேவை வழக்கத்தை ஒட்டியே இருக்கும் என எதிர்பார்க்கப்படுகிறது. குறைந்தபட்ச முன்னறிவிப்பு {time} அளவில் {v:.2f} kWh.",
        "actual_vs_pred": "உண்மையான மற்றும் கணிக்கப்பட்ட நுகர்வு",
        "future_forecast": "எதிர்கால தேவை முன்னறிவிப்பு",
        "hourly_schedule": "மணிநேர முன்னறிவிப்பு அட்டவணை",
        "col_time": "நேரம்",
        "col_expected": "எதிர்பார்க்கும் பயன்பாடு",
        "col_timestamp": "நேரமுத்திரை",
        "col_temperature": "வெப்பநிலை",
        "col_humidity": "ஈரப்பதம்",
        "col_wind": "காற்று வேகம்",
        "col_consumption": "நுகர்வு",
        "col_saved": "சேமித்தது",
        "normal": "இயல்பானது",
        "download_forecast": "முன்னறிவிப்பு CSV பதிவிறக்கு",
        "download_saved": "சேமித்த பதிவுகள் CSV பதிவிறக்கு",
        "sqlite_info_help": "இங்கே என்ன சேமிக்கப்படுகிறது?",
        "f_date": "தேதி",
        "f_time": "நேரம்",
        "f_consumption": "நுகர்வு (kWh)",
        "save_reading": "அளவீட்டை SQLite-இல் சேமி",
        "reading_saved": "அளவீடு SQLite தரவுத்தளத்தில் சேமிக்கப்பட்டது.",
        "no_readings": "சேமித்த அளவீடுகள் இன்னும் இல்லை. நேரடி அளவீட்டு பொத்தானைப் பயன்படுத்தவும் அல்லது தற்போதைய தரவைச் சேமிக்கவும்.",
        "units_title": "யூனிட்களும் மதிப்பிடப்பட்ட கட்டணமும்",
        "units_note": "1 யூனிட் = 1 kWh மின்சாரம். கட்டணங்கள் 60 நாள் (இரு மாத) காலத்துக்கான TN EB வீட்டு மின் கட்டண ஸ்லாப்களைக் கொண்டு மதிப்பிடப்படுகின்றன.",
        "units_two_month": "{u:,.0f} யூனிட் / 2 மாதங்கள்",
        "units_per_day": "நாளொன்றுக்கு {u:.1f} யூனிட்",
        "units_bill": "மதிப்பிடப்பட்ட கட்டணம்: ரூ. {b:,.0f}",
        "units_below": "ரூ. {limit:,}-க்குக் கீழ்",
        "units_above": "ரூ. {limit:,}-க்கு மேல்",
        "units_selected": "தேர்ந்தெடுத்தது",
        "units_dataset_total": "தரவுத்தொகுப்பு மொத்தம்: {d:.0f} நாட்களில் {u:,.0f} யூனிட்",
        "units_sidebar": "≈ {u:,.0f} யூனிட் / 2 மாதங்கள் · ரூ. {b:,.0f}",
        "units_help": "ஒரு அளவீட்டில் பயன்படுத்திய சராசரி மின்சாரம். 1 யூனிட் = 1 kWh.",
        "dlg_reg_title": "கணக்கு உருவாக்கப்பட்டது!",
        "dlg_reg_welcome": "வரவேற்கிறோம், {name}!",
        "dlg_reg_body": "{email} க்கான உங்கள் கணக்கு தயார்.",
        "dlg_reg_next": "அடுத்த படி: **{login}** தாவலைத் திறந்து உங்கள் மின்னஞ்சல் மற்றும் கடவுச்சொல்லுடன் உள்நுழையவும்.",
        "dlg_reg_privacy": "உங்கள் அளவீடுகள் உங்கள் சொந்த SQLite தரவுத்தளத்தில் தனிப்பட்டதாக இருக்கும். ஒவ்வொரு கணக்கிலும் அதிகபட்சம் {max:,} அளவீடுகள் வைக்கப்படும்.",
        "reg_success_line": "கணக்கு உருவாக்கப்பட்டது. இப்போது உள்நுழையலாம்.",
        "warn_capped": "தரவுத்தொகுப்பு சமீபத்திய {max:,} அளவீடுகளுக்கு வரையறுக்கப்பட்டுள்ளது — {dropped:,} பழைய அளவீடுகள் விடப்பட்டன.",
        "toast_capped": "சமீபத்திய {max:,} அளவீடுகள் மட்டுமே பயன்படுத்தப்படுகின்றன.",
        "upload_limit_hint": "தரவுத்தொகுப்புகள் {max:,} அளவீடுகளுக்கு வரையறுக்கப்பட்டுள்ளன. உங்கள் கோப்பில் அதிகம் இருந்தால், சமீபத்திய {max:,} மட்டுமே பயன்படுத்தப்படும்.",
        "dlg_sqlite_limit": "- **வரம்பு:** ஒவ்வொரு கணக்கிலும் அதிகபட்சம் {max:,} அளவீடுகள் வைக்கப்படும்; வரம்பை அடைந்தால் பழையவை தானாக நீக்கப்படும்.",
        "bd_range": "வரம்பு",
        "bd_avg": "சராசரி",
        "bd_energy_share": "ஆற்றல் பங்கு",
        "bd_peak_time": "பொதுவான நேரம்",
        "bd_avg_temp": "சராசரி வெப்பநிலை",
        "bd_bar_readings": "அளவீடுகளின் பங்கு",
        "bd_bar_energy": "ஆற்றலின் பங்கு (kWh)",
        "period_night": "இரவு (12–6 AM)",
        "period_morning": "காலை (6 AM–12 PM)",
        "period_afternoon": "மதியம் (12–6 PM)",
        "period_evening": "மாலை (6 PM–12 AM)",
        "period_short_night": "இரவு",
        "period_short_morning": "காலை",
        "period_short_afternoon": "மதியம்",
        "period_short_evening": "மாலை",
        "bd_insight_energy": "‘{level}’ அளவீடுகள் மொத்த அளவீடுகளில் {rp:.0f}% மட்டுமே, ஆனால் {ep:.0f}% ஆற்றலைப் பயன்படுத்துகின்றன.",
        "bd_insight_when": "பெரும்பாலான ‘{level}’ அளவீடுகள் ({p:.0f}%) {period} நேரத்தில் நிகழ்கின்றன.",
        "bd_insight_weather": "‘{hi_label}’ அளவீடுகளின் சராசரி வெப்பநிலை {hot:.1f} °C; ‘{lo_label}’ அளவீடுகளுக்கு {cool:.1f} °C.",
        "bd_tab_hour": "மணிநேரவாரி",
        "bd_tab_period": "நாளின் நேரம்",
        "bd_tab_day": "வேலை நாள் – வார இறுதி",
        "bd_tab_weather": "வானிலை",
        "bd_hour_caption": "நாளின் ஒவ்வொரு மணிநேரத்திலும் ஒவ்வொரு நிலையிலும் எத்தனை அளவீடுகள் உள்ளன.",
        "bd_period_caption": "நாளின் ஒவ்வொரு பகுதியிலும் குறைவு, நடுத்தரம், அதிகம் என உள்ள அளவீடுகளின் சதவீதம்.",
        "bd_day_caption": "வேலை நாள், வார இறுதி அளவீடுகள் ஒவ்வொரு நிலையிலும் எத்தனை சதவீதம்.",
        "bd_weather_caption": "ஒவ்வொரு நிலையிலும் உள்ள அளவீடுகளின் சராசரி பயன்பாடும் வானிலையும்.",
        "col_period": "நேரப் பிரிவு",
        "col_daytype": "நாள் வகை",
        "col_level": "நிலை",
        "col_avg_kwh": "சராசரி பயன்பாடு (kWh)",
        "col_readings": "அளவீடுகள்",
        "col_avg_temp": "சராசரி வெப்பநிலை (°C)",
        "col_avg_hum": "சராசரி ஈரப்பதம் (%)",
        "col_avg_wind": "சராசரி காற்று (km/h)",
        "bd_weekday": "வேலை நாள்",
        "bd_weekend": "வார இறுதி",
        "tips_title": "ஆற்றலை சேமிக்க பரிந்துரைகள்",
        "tips_caption": "இந்தத் தரவில் உள்ள முறைகளின் அடிப்படையில் சில நடைமுறை யோசனைகள்.",
        "tip_peak_shift_title": "உச்ச நேரத்திலிருந்து பணிகளை மாற்றவும்",
        "tip_peak_shift_detail": "{peak} அளவில் பயன்பாடு அதிகமாக உள்ளது. சலவை இயந்திரம், நீர் ஹீட்டர் போன்றவற்றை {off} நேரத்தில் இயக்கினால் உச்ச தேவையையும் கட்டணத்தையும் குறைக்கலாம்.",
        "tip_cooling_title": "வெப்பமான நேரங்களில் குளிரூட்டலைக் குறைக்கவும்",
        "tip_cooling_detail": "இந்தத் தரவில் வெப்பம் அதிகரிக்கும்போது பயன்பாடு கணிசமாக உயர்கிறது. ஏசியை சில டிகிரி அதிகமாக அமைத்து, மின்விசிறியுடன் இணைத்துப் பயன்படுத்தி, நேரடி வெயிலில் திரைச்சீலைகளை மூடுவது குளிரூட்டல் செலவைக் குறைக்கும்.",
        "tip_weekend_high_title": "வார இறுதியில் பயன்பாடு அதிகம்",
        "tip_weekend_high_detail": "வார இறுதி நாட்களில் அளவீடுகள் வேலை நாட்களை விட அதிகம். கனமான வீட்டு வேலைகளை (சலவை, சமையல்) வாரம் முழுவதும் பரப்புவது தேவையை சீராக்கும்.",
        "tip_weekday_high_title": "வேலை நாட்களில் பயன்பாடு அதிகம்",
        "tip_weekday_high_detail": "வேலை நாட்களில் அளவீடுகள் வார இறுதியை விட அதிகம். வேலை நேரத்தில் இயங்கிக்கொண்டிருக்கும் சாதனங்களால் இது ஏற்பட்டால், ஒரு டைமர் அல்லது ஸ்மார்ட் பிளக் உதவும்.",
        "tip_high_share_title": "பெரும்பாலான அளவீடுகள் அதிகம் எனக் காட்டுகின்றன",
        "tip_high_share_detail": "{pct:.0f}% அளவீடுகள் அதிக நிலையில் உள்ளன. எப்போதும் இயங்கும் ஒரு கனமான சாதனத்தை (பழைய குளிர்சாதனப்பெட்டி, நீர் ஹீட்டர், ஏசி) சரிபார்ப்பது இதைக் குறைக்க உதவும்.",
        "tip_city_title": "பெருநகர அளவிலான குளிரூட்டலும் வணிகச் சுமையும்",
        "tip_city_detail": "பெருநகர பயன்பாடு பெரும்பாலும் குளிரூட்டல் மற்றும் வணிக நடவடிக்கையால் இயக்கப்படுகிறது. ஏசிக்கு டைமர் அமைப்பதும், 5-நட்சத்திர திறன்மிக்க சாதனங்களைத் தேர்ந்தெடுப்பதும் இங்கு அதிகம் உதவும்.",
        "tip_village_title": "குறைந்த, சீரான அடிப்படை அளவைத் தக்கவைக்கவும்",
        "tip_village_detail": "கிராம பயன்பாடு ஏற்கெனவே குறைவாகவும் சீராகவும் உள்ளது. கனமான குளிரூட்டல் சுமை அதிகம் இல்லாததால், விளக்குகளை LED ஆக மாற்றுவதும், ஸ்டாண்ட்பை மின் பயன்பாட்டைத் தவிர்ப்பதும் மிகப்பெரிய மேம்பாட்டைத் தரும்.",
        "tip_led_title": "LED விளக்குகளுக்கு மாறவும்",
        "tip_led_detail": "அதே பிரகாசத்திற்கு LED விளக்குகள் பழைய பல்புகளை விட சுமார் 75-80% குறைவான மின்சாரத்தைப் பயன்படுத்தும், மேலும் நீண்ட காலம் நீடிக்கும்.",
        "tip_standby_title": "பயன்படுத்தாத மின்சாதனங்களை அவிழ்க்கவும்",
        "tip_standby_detail": "டிவி, சார்ஜர்கள், செட் டாப் பாக்ஸ் போன்றவை பிளக் செய்யப்பட்டே இருந்தால் சிறிதளவு ஸ்டாண்ட்பை மின்சாரத்தைத் தொடர்ந்து இழுக்கும். இரவில் அணைக்கும் ஒரு பவர் ஸ்ட்ரிப் மாதம் முழுவதும் சேமிப்பைத் தரும்.",
        "room_usage_title": "அறை வாரியான பயன்பாடு",
        "room_usage_caption": "உங்கள் உண்மையான பயன்பாட்டில் பொதுவான வீட்டு பிரிவைப் பயன்படுத்தி, அது எங்கு செல்கிறது என்பதற்கான தோராயமான வழிகாட்டி.",
        "room_usage_disclaimer": "பொதுவான பயன்பாட்டு முறைகளிலிருந்து மதிப்பிடப்பட்டது, ஒவ்வொரு அறையிலும் அளக்கப்படவில்லை — இந்த ஆப்பில் அறை வாரியான சென்சார்கள் இல்லை.",
        "room_est_cost": "≈ ரூ. {v:,.0f}",
        "room_kitchen": "சமையலறை",
        "room_living": "வரவேற்பறை & பொழுதுபோக்கு",
        "room_bedroom": "படுக்கையறை & ஏசி",
        "room_water_heater": "நீர் ஹீட்டர்",
        "room_lighting_other": "விளக்குகள் & பிற",
        "reasons_title": "பயன்பாடு ஏன் {level}?",
        "reasons_forecast_title": "{time} அளவில் தேவை ஏன் உச்சத்தை அடைகிறது?",
        "reasons_caption": "இந்தத் தரவுத்தொகுப்பில் உள்ள அளவீடுகளிலிருந்து கணக்கிடப்பட்ட இந்த நிலைக்கான காரணங்கள். வெளிக்கோடிட்ட அட்டை முக்கிய காரணம்.",
        "reasons_summary_above": "{v:.2f} kWh என்பது சராசரியான {avg:.2f} kWh-ஐ விட {p:.0f}% அதிகம்.",
        "reasons_summary_below": "{v:.2f} kWh என்பது சராசரியான {avg:.2f} kWh-ஐ விட {p:.0f}% குறைவு.",
        "reasons_summary_near": "{v:.2f} kWh என்பது சராசரியான {avg:.2f} kWh-க்கு அருகில் உள்ளது.",
        "tag_raises": "பயன்பாட்டை உயர்த்துகிறது",
        "tag_lowers": "பயன்பாட்டைக் குறைக்கிறது",
        "tag_neutral": "பெரிய தாக்கம் இல்லை",
        "tag_context": "பின்னணி",
        "tag_main": "முக்கிய காரணம்",
        "est_effect": " (இந்தத் தரவில் சுமார் {e:+.2f} kWh)",
        "reason_peak_hour_title": "உச்ச தேவை நேரம்",
        "reason_peak_hour_detail": "இந்தத் தரவில் {hour} மிகவும் பரபரப்பான நேரங்களில் ஒன்று — இங்கு சராசரி பயன்பாடு {v:.2f} kWh, தினசரி சராசரியை விட {p:.0f}% அதிகம்.",
        "reason_offpeak_hour_title": "அமைதியான நேரம்",
        "reason_offpeak_hour_detail": "இந்தத் தரவில் {hour} மிகவும் அமைதியான நேரங்களில் ஒன்று — இங்கு சராசரி பயன்பாடு {v:.2f} kWh, தினசரி சராசரியை விட {p:.0f}% குறைவு.",
        "reason_normal_hour_title": "வழக்கமான நேரம்",
        "reason_normal_hour_detail": "{hour} ஒரு சாதாரண நேரம்; இங்கு சராசரி பயன்பாடு {v:.2f} kWh.",
        "reason_hot_title": "வெப்பமான வானிலை — அதிக குளிரூட்டல்",
        "reason_hot_detail": "{temp:.1f} °C என்பது வழக்கமான {avg:.1f} °C-ஐ விட {d:.1f} °C அதிகம்; எனவே மின்விசிறிகளும் குளிரூட்டிகளும் கூடுதலாக இயங்குகின்றன{est}.",
        "reason_cool_title": "குளிர்ந்த வானிலை — குறைந்த குளிரூட்டல்",
        "reason_cool_detail": "{temp:.1f} °C என்பது வழக்கமான {avg:.1f} °C-ஐ விட {d:.1f} °C குறைவு; எனவே குளிரூட்டல் தேவை குறைவு{est}.",
        "reason_temp_normal_title": "இயல்பான வெப்பநிலை",
        "reason_temp_normal_detail": "{temp:.1f} °C என்பது வழக்கமான {avg:.1f} °C-க்கு அருகில் உள்ளது; எனவே வானிலை பயன்பாட்டை எந்தப் பக்கமும் தள்ளவில்லை.",
        "reason_weekend_title": "வார இறுதி முறை",
        "reason_weekend_detail": "இது வார இறுதி நாள். இந்தத் தரவில் வார இறுதிகளின் சராசரி {w:.2f} kWh, வேலை நாட்களில் {d:.2f} kWh.",
        "reason_weekday_title": "வேலை நாள் முறை",
        "reason_weekday_detail": "இது வேலை நாள். இந்தத் தரவில் வேலை நாட்களின் சராசரி {d:.2f} kWh, வார இறுதிகளில் {w:.2f} kWh.",
        "reason_humid_title": "அதிக ஈரப்பதம்",
        "reason_humid_detail": "{h:.0f}% ஈரப்பதம் அதிக வெப்பமாக உணர வைப்பதால், மின்விசிறிகளும் குளிரூட்டிகளும் நீண்ட நேரம் இயங்க வாய்ப்பு அதிகம்.",
        "reason_windy_title": "வலுவான காற்று",
        "reason_windy_detail": "{w:.1f} km/h காற்று இயற்கை காற்றோட்டத்தைத் தருவதால், குளிரூட்டல் தேவை குறைகிறது.",
        "reason_none_title": "வழக்கமான முறையைப் பின்பற்றுகிறது",
        "reason_none_detail": "தனித்துத் தெரியும் வலுவான காரணம் எதுவும் இல்லை; பயன்பாடு வழக்கமான வழக்கத்தைப் பின்பற்றுகிறது.",
        "reason_loc_title": "இடச் சுயவிவரம்: {loc}",
        "reason_loc_village": "கிராம வீடுகளில் கனமான சாதனங்கள் குறைவு; அதனால் அடிப்படைச் சுமை குறைவாகவும் சீராகவும் இருக்கும்.",
        "reason_loc_town": "நகரங்களில் குடும்பங்களும் சிறு கடைகளும் அன்றாட வழக்கத்தைப் பின்பற்றுவதால் காலை, மாலை உச்சங்கள் தெளிவாகத் தெரியும்.",
        "reason_loc_city": "பெருநகரங்களில் அதிக குளிரூட்டல் மற்றும் வணிகச் சுமை இருப்பதால் அடிப்படை அளவு உயர்ந்து, பரபரப்பு நேர உச்சங்கள் கூர்மையாகும்.",
        "low": "குறைவு", "medium": "நடுத்தரம்", "high": "அதிகம்", "saved_records": "சேமிக்கப்பட்ட பதிவுகள்",
    },
}


def t(key: str, **kwargs: object) -> str:
    """Translate a UI string key into the currently selected language, falling
    back to English (or the raw key) if a translation is missing."""
    language = st.session_state.get("lang", "English")
    catalog = TRANSLATIONS.get(language, TRANSLATIONS["English"])
    text = catalog.get(key) or TRANSLATIONS["English"].get(key, key)
    return text.format(**kwargs) if kwargs else text


THEME_PALETTES: dict[str, dict[str, str]] = {
    "Light": {
        "app_bg": "#f4f6fb", "card_bg": "#ffffff", "card_border": "#e5e7eb",
        "card_border_hover": "#c7d2fe", "card_shadow": "rgba(15,23,42,0.04)",
        "card_shadow_hover": "rgba(15,23,42,0.07)", "text_primary": "#1e293b",
        "text_secondary": "#475569", "text_muted": "#64748b", "sidebar_bg": "#ffffff",
        "sidebar_border": "#e2e8f0", "input_bg": "#ffffff", "input_border": "#d1d5db",
        "input_text": "#1e293b", "placeholder": "#94a3b8", "accent": "#4f46e5",
        "accent_hover": "#4338ca", "accent_soft": "rgba(79,70,229,0.15)",
        "segctl_bg": "#f1f5f9", "segctl_border": "#e5e7eb", "segctl_text": "#334155",
        "tab_border": "#e5e7eb", "tab_text": "#475569", "btn_secondary_text": "#334155",
        "dialog_shadow": "rgba(15,23,42,0.25)", "toggle_off": "#cbd5e1",
        "icon_dashboard": "#e0e7ff", "icon_forecast": "#ede9fe",
        "icon_data": "#fef3c7", "icon_sqlite": "#ccfbf1",
    },
    "Dark": {
        "app_bg": "#0b1120", "card_bg": "#161d31", "card_border": "rgba(255,255,255,0.08)",
        "card_border_hover": "rgba(129,140,248,0.45)", "card_shadow": "rgba(0,0,0,0.35)",
        "card_shadow_hover": "rgba(0,0,0,0.5)", "text_primary": "#f1f5f9",
        "text_secondary": "#cbd5e1", "text_muted": "#94a3b8", "sidebar_bg": "#0f1526",
        "sidebar_border": "rgba(255,255,255,0.08)", "input_bg": "#1b2338",
        "input_border": "rgba(255,255,255,0.18)", "input_text": "#f1f5f9",
        "placeholder": "#8590a6", "accent": "#818cf8", "accent_hover": "#6366f1",
        "accent_soft": "rgba(129,140,248,0.22)", "segctl_bg": "#1b2338",
        "segctl_border": "rgba(255,255,255,0.10)", "segctl_text": "#cbd5e1",
        "tab_border": "rgba(255,255,255,0.10)", "tab_text": "#94a3b8",
        "btn_secondary_text": "#e2e8f0", "dialog_shadow": "rgba(0,0,0,0.6)",
        "toggle_off": "#3f4b64",
        "icon_dashboard": "rgba(79,70,229,0.28)", "icon_forecast": "rgba(124,58,237,0.28)",
        "icon_data": "rgba(217,119,6,0.28)", "icon_sqlite": "rgba(13,148,136,0.28)",
    },
}


def add_transition_styles() -> None:
    """Professional theme with a light/dark switch: cards, indigo accents, restrained motion."""
    p = THEME_PALETTES.get(st.session_state.get("theme", "Light"), THEME_PALETTES["Light"])
    st.html(
        f"""
        <style>
        /* =====================================================================
           Motion: a gentle fade/rise on load, kept subtle and reduced-motion safe.
           ===================================================================== */
        @keyframes energysmart-reveal {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stMetric"] {{
            animation: energysmart-reveal 360ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        @keyframes energysmart-page-enter {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        [class*="st-key-view-"] {{
            animation: energysmart-page-enter 360ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        [class*="st-key-view-"] [data-testid="stVerticalBlockBorderWrapper"] {{
            animation: energysmart-reveal 360ms cubic-bezier(0.22, 1, 0.36, 1) both;
            animation-delay: 40ms;
        }}
        [data-testid="stTabContent"] {{
            animation: energysmart-page-enter 320ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        @media (prefers-reduced-motion: reduce) {{
            [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"],
            [class*="st-key-view-"], [data-testid="stTabContent"] {{ animation: none !important; }}
        }}

        /* =====================================================================
           Layout & responsiveness
           ===================================================================== */
        .block-container {{ padding-top: 1.75rem; padding-left: 1.25rem; padding-right: 1.25rem; max-width: 100%; }}
        [data-testid="stMetric"] {{ min-width: 140px; }}
        .info-btn button {{ padding: 0.25rem 0.6rem !important; border-radius: 999px !important; }}

        @media (max-width: 768px) {{
            .block-container {{ padding-left: 0.75rem !important; padding-right: 0.75rem !important; padding-top: 1rem !important; }}
            div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; row-gap: 0.6rem !important; }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                min-width: 100% !important; width: 100% !important; flex: 1 1 100% !important;
            }}
            [data-testid="stMetricValue"] {{ font-size: 1.4rem !important; }}
            h1 {{ font-size: 1.5rem !important; }}
            h2, h3 {{ font-size: 1.1rem !important; }}
            [data-testid="stDataFrame"] {{ width: 100% !important; }}
            div[data-testid="stDialog"] > div {{ width: 92vw !important; }}
        }}
        @media (max-width: 480px) {{
            [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }}
            .stButton button {{ width: 100% !important; }}
        }}

        /* =====================================================================
           Theme: clean background, cards, indigo accents. Colors come from the
           THEME_PALETTES dict above, so Light/Dark share one set of rules.
           Native widgets (charts, dataframes, alerts) are deliberately left as
           light panels in both modes since they can't be recolored via CSS.
           ===================================================================== */
        .stApp {{ background: {p['app_bg']} !important; }}
        .stApp [data-testid="stHeader"] {{ background: transparent !important; }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {p['text_primary']} !important; font-weight: 700 !important;
        }}
        .stApp [data-testid="stCaptionContainer"] {{ color: {p['text_secondary']} !important; font-weight: 500 !important; }}
        .stApp [data-testid="stWidgetLabel"] p {{ color: {p['text_secondary']} !important; font-weight: 600 !important; }}
        .stApp a {{ color: {p['accent']} !important; }}
        .stApp a:hover {{ color: {p['accent_hover']} !important; }}
        .stApp strong {{ color: {p['text_primary']} !important; }}
        .stApp p, .stApp li, .stApp span {{ color: {p['text_primary']} !important; }}

        /* Sidebar */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {{
            background: {p['sidebar_bg']} !important;
        }}
        [data-testid="stSidebar"] {{ border-right: 1px solid {p['sidebar_border']} !important; }}

        /* Cards: every bordered container becomes a clean panel */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {p['card_bg']} !important;
            border: 1px solid {p['card_border']} !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 2px {p['card_shadow']}, 0 8px 20px {p['card_shadow']};
            transition: box-shadow 200ms ease, border-color 200ms ease;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: {p['card_border_hover']} !important;
            box-shadow: 0 4px 10px {p['card_shadow_hover']}, 0 12px 28px {p['card_shadow_hover']};
        }}

        /* Metrics: card, bold value, small colored top accent per view (see below) */
        [data-testid="stMetric"] {{
            background: {p['card_bg']}; border: 1px solid {p['card_border']}; border-radius: 12px; padding: 0.85rem 1rem !important;
        }}
        [data-testid="stMetricValue"] {{ color: {p['text_primary']} !important; font-weight: 700 !important; }}
        [data-testid="stMetricLabel"] {{ color: {p['text_secondary']} !important; font-weight: 600 !important; }}

        /* Buttons */
        .stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {{
            border-radius: 10px !important; font-weight: 600 !important;
            transition: transform 150ms ease, box-shadow 150ms ease, background-color 150ms ease !important;
        }}
        .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
            background: {p['accent']} !important; border: none !important; color: #ffffff !important;
            box-shadow: 0 1px 2px {p['accent_soft']}, 0 6px 16px {p['accent_soft']} !important;
        }}
        .stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {{
            background: {p['accent_hover']} !important; transform: translateY(-1px) !important;
        }}
        .stButton button[kind="secondary"], [data-testid="stDownloadButton"] button {{
            background: {p['card_bg']} !important; border: 1px solid {p['input_border']} !important; color: {p['btn_secondary_text']} !important;
        }}
        .stButton button[kind="secondary"]:hover, [data-testid="stDownloadButton"] button:hover {{
            border-color: {p['accent']} !important; color: {p['accent']} !important; transform: translateY(-1px) !important;
        }}

        /* Tabs: clean underline style */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 1.25rem; background: transparent; border-bottom: 1px solid {p['tab_border']}; border-radius: 0;
        }}
        [data-testid="stTabs"] button[data-baseweb="tab"] {{
            color: {p['tab_text']} !important; font-weight: 600 !important; transition: color 180ms ease !important;
        }}
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{ color: {p['accent']} !important; }}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: {p['accent']} !important; height: 2px !important; }}

        /* Segmented control: solid indigo pill for the selected option */
        div[data-testid="stSegmentedControl"] {{
            gap: 0.3rem; background: {p['segctl_bg']}; padding: 0.25rem; border-radius: 999px;
            border: 1px solid {p['segctl_border']}; width: fit-content;
        }}
        div[data-testid="stSegmentedControl"] label {{
            background: transparent !important; color: {p['segctl_text']} !important; border-radius: 999px !important;
            font-weight: 600 !important; transition: background-color 180ms ease, color 180ms ease, transform 150ms ease;
        }}
        div[data-testid="stSegmentedControl"] label p {{ color: inherit !important; }}
        div[data-testid="stSegmentedControl"] label:active {{ transform: scale(0.97); }}
        div[data-testid="stSegmentedControl"] label[aria-checked="true"] {{ background: {p['accent']} !important; }}
        div[data-testid="stSegmentedControl"] label[aria-checked="true"] p {{ color: #ffffff !important; }}

        /* Workspace nav: each tab's selected pill takes on that tab's own accent
           color (indigo / violet / amber / teal) instead of one uniform color. */
        [class*="st-key-workspace_view"] label:nth-child(1)[aria-checked="true"] {{ background: #4f46e5 !important; }}
        [class*="st-key-workspace_view"] label:nth-child(2)[aria-checked="true"] {{ background: #7c3aed !important; }}
        [class*="st-key-workspace_view"] label:nth-child(3)[aria-checked="true"] {{ background: #d97706 !important; }}
        [class*="st-key-workspace_view"] label:nth-child(4)[aria-checked="true"] {{ background: #0d9488 !important; }}

        /* Inputs */
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input,
        [data-testid="stFileUploader"] section {{
            background: {p['input_bg']} !important; border: 1px solid {p['input_border']} !important;
            border-radius: 10px !important; color: {p['input_text']} !important; caret-color: {p['input_text']} !important;
            transition: box-shadow 180ms ease, border-color 180ms ease !important;
        }}
        [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus,
        [data-testid="stDateInput"] input:focus, [data-testid="stTimeInput"] input:focus {{
            border-color: {p['accent']} !important; box-shadow: 0 0 0 3px {p['accent_soft']} !important;
        }}
        [data-testid="stNumberInput"] button, [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{
            background: {p['card_bg']} !important; border: 1px solid {p['input_border']} !important; color: {p['btn_secondary_text']} !important;
        }}
        [data-testid="stFileUploader"] section button {{
            background: {p['card_bg']} !important; border: 1px solid {p['input_border']} !important; color: {p['btn_secondary_text']} !important;
        }}
        [data-testid="stToggle"] div[role="switch"] {{ background: {p['toggle_off']} !important; }}
        [data-testid="stToggle"] div[role="switch"][aria-checked="true"] {{ background: {p['accent']} !important; }}
        .stApp input::placeholder, .stApp textarea::placeholder {{ color: {p['placeholder']} !important; opacity: 1 !important; }}

        /* Alerts, dataframes, dialogs, expanders: native components are kept as
           light panels in both themes (they can't be recolored via CSS), just
           rounded to match the rest of the app. */
        [data-testid="stAlert"] {{ border-radius: 12px !important; }}
        [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid {p['card_border']}; }}
        div[data-testid="stDialog"] > div {{
            background: {p['card_bg']} !important; border-radius: 16px !important;
            box-shadow: 0 20px 60px {p['dialog_shadow']} !important;
        }}
        div[data-testid="stDialog"] * {{ color: {p['text_primary']} !important; }}
        div[data-testid="stDialog"] [data-testid="stCaptionContainer"] {{ color: {p['text_secondary']} !important; }}
        [data-testid="stExpander"] {{ background: {p['card_bg']} !important; border: 1px solid {p['card_border']} !important; border-radius: 12px !important; }}

        /* =====================================================================
           Per-tab section header: a clean card with a colored icon badge —
           no gradients or motion, just enough color to tell the tabs apart.
           ===================================================================== */
        .view-banner {{
            display: flex; align-items: center; gap: 0.9rem;
            background: {p['card_bg']}; border: 1px solid {p['card_border']}; border-radius: 14px;
            padding: 1rem 1.25rem; margin-bottom: 1.1rem;
            box-shadow: 0 1px 2px {p['card_shadow']};
        }}
        .view-banner-icon {{
            font-size: 1.5rem; width: 46px; height: 46px; border-radius: 12px; flex: 0 0 auto;
            display: flex; align-items: center; justify-content: center;
        }}
        .view-banner-title {{ font-size: 1.15rem; font-weight: 700; color: {p['text_primary']} !important; }}
        .view-banner-subtitle {{ font-size: 0.86rem; color: {p['text_secondary']} !important; }}

        .view-banner-dashboard .view-banner-icon {{ background: {p['icon_dashboard']}; }}
        .view-banner-forecast .view-banner-icon {{ background: {p['icon_forecast']}; }}
        .view-banner-data .view-banner-icon {{ background: {p['icon_data']}; }}
        .view-banner-sqlite .view-banner-icon {{ background: {p['icon_sqlite']}; }}

        [class*="st-key-view-dashboard"] [data-testid="stMetric"] {{ border-top: 3px solid #4f46e5; }}
        [class*="st-key-view-forecast"] [data-testid="stMetric"] {{ border-top: 3px solid #7c3aed; }}
        [class*="st-key-view-data"] [data-testid="stMetric"] {{ border-top: 3px solid #d97706; }}
        [class*="st-key-view-sqlite-database"] [data-testid="stMetric"] {{ border-top: 3px solid #0d9488; }}

        /* Theme switch buttons */
        [class*="st-key-theme_light"] button, [class*="st-key-theme_dark"] button {{
            padding: 0.25rem 0.7rem !important; font-size: 0.85rem !important;
        }}

        @media (prefers-reduced-motion: reduce) {{
            [data-testid="stVerticalBlockBorderWrapper"],
            .stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {{
                transition: none !important;
            }}
        }}
        </style>
        """
    )


add_transition_styles()


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA)


LOCATION_PROFILES = {
    "Village": {"base": 1.0, "peak_amp": 0.5, "noise": 0.15, "ac_sensitivity": 0.015, "commercial": 0.05, "heat": 0.0, "target_kwh_per_hour": 0.30},
    "Town": {"base": 2.3, "peak_amp": 1.1, "noise": 0.30, "ac_sensitivity": 0.035, "commercial": 0.20, "heat": 1.0, "target_kwh_per_hour": 0.75},
    "City": {"base": 4.5, "peak_amp": 2.2, "noise": 0.55, "ac_sensitivity": 0.065, "commercial": 0.45, "heat": 2.5, "target_kwh_per_hour": 1.20},
}


LOCATION_SEEDS = {"Village": 101, "Town": 202, "City": 303}


def _max_units_under_bill(limit: float) -> float:
    """Largest two-month unit count whose TN EB bill stays at or below `limit` (bisection)."""
    low, high = 0.0, 1_000_000.0
    for _ in range(60):
        mid = (low + high) / 2
        if calculate_tn_eb_domestic_bill(mid)[0] <= limit:
            low = mid
        else:
            high = mid
    return low


def _keep_bill_under_limit(consumption: np.ndarray) -> np.ndarray:
    """Safety net: scale a sample down if its projected two-month bill would reach SAMPLE_BILL_LIMIT."""
    projected_units = float(consumption.mean()) * 24 * 60
    max_units = _max_units_under_bill(SAMPLE_BILL_LIMIT * 0.95)
    if projected_units <= 0 or projected_units <= max_units:
        return consumption
    return consumption * (max_units / projected_units)


@st.cache_data(show_spinner=False)
def generate_sample_data(location_type: str, periods: int = 24 * 45) -> pd.DataFrame:
    """Synthesize an hourly electricity-usage sample dataset for a village, town,
    or city. Each location type has its own realistic demand profile: villages
    run low and flat, towns show clear morning/evening peaks, and cities layer
    heavier cooling load and commercial activity on top of sharper peaks."""
    periods = min(int(periods), MAX_DATASET_ROWS)
    profile = LOCATION_PROFILES.get(location_type, LOCATION_PROFILES["Town"])
    rng = np.random.default_rng(seed=LOCATION_SEEDS.get(location_type, 0))

    end = pd.Timestamp.now().floor("h")
    timestamps = pd.date_range(end=end, periods=periods, freq="h")
    hour = np.asarray(timestamps.hour)
    day_of_year = np.asarray(timestamps.dayofyear)
    is_weekend = np.asarray(timestamps.weekday >= 5)

    # Weather: seasonal + daily temperature swing, with a mild urban-heat bump for towns/cities.
    season_temp = 27 + 6 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_temp_swing = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = season_temp + daily_temp_swing + profile["heat"] + rng.normal(0, 1.2, periods)
    humidity = np.clip(70 - 0.6 * (temperature - 27) + rng.normal(0, 5, periods), 30, 95)
    wind_speed = np.clip(6 + 3 * np.sin(2 * np.pi * hour / 24 + 1) + rng.normal(0, 1.5, periods), 0, None)

    # Demand shape: morning + evening peaks, sharper in cities, flatter in villages.
    peak_width = 1.6 if location_type == "City" else 2.4
    morning_peak = np.exp(-0.5 * ((hour - 8) / peak_width) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 19.5) / (peak_width + 0.3)) ** 2)
    daily_shape = profile["peak_amp"] * (morning_peak + evening_peak)

    weekend_multiplier = np.where(is_weekend, 1.08 if location_type == "Village" else 0.95, 1.0)
    cooling_load = profile["ac_sensitivity"] * np.clip(temperature - 26, 0, None) ** 1.4
    commercial_load = profile["commercial"] * np.where((hour >= 9) & (hour <= 20) & ~is_weekend, 1.0, 0.25)
    noise = rng.normal(0, profile["noise"], periods)

    consumption = (profile["base"] + daily_shape + cooling_load + commercial_load) * weekend_multiplier + noise
    consumption = np.clip(consumption, 0.1, None)
    # Keep the shape of the demand curve but rescale to a realistic average, so the projected
    # two-month bill stays comfortably below SAMPLE_BILL_LIMIT (Rs. 20,000).
    consumption = consumption * (profile["target_kwh_per_hour"] / float(consumption.mean()))
    consumption = np.clip(_keep_bill_under_limit(consumption), 0.02, None)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": np.round(temperature, 1),
            "humidity": np.round(humidity, 1),
            "wind_speed": np.round(wind_speed, 1),
            "consumption_kwh": np.round(consumption, 3),
        }
    )


@st.cache_resource(show_spinner=False)
def train_cached_model(data: pd.DataFrame):
    return train_model(data)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def calculate_tn_eb_domestic_bill(units: float) -> tuple[float, float]:
    """Estimate Tamil Nadu domestic LT-IA energy charges for a two-month bill."""
    consumption = max(0.0, float(units))
    if consumption <= 500:
        free_units = min(consumption, 200.0)
        bill = min(max(consumption - 200, 0), 200) * 4.70
        bill += min(max(consumption - 400, 0), 100) * 6.30
        return round(bill, 2), free_units

    free_units = 100.0
    slabs = [
        (300, 4.70),
        (100, 6.30),
        (100, 8.40),
        (200, 9.45),
        (200, 10.50),
        (float("inf"), 11.55),
    ]
    remaining_units = consumption - free_units
    bill = 0.0
    for slab_units, rate in slabs:
        units_in_slab = min(remaining_units, slab_units)
        bill += units_in_slab * rate
        remaining_units -= units_in_slab
        if remaining_units <= 0:
            break
    return round(bill, 2), free_units


def init_energy_table() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS energy_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                wind_speed REAL NOT NULL,
                consumption_kwh REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, timestamp)
            )
            """
        )


def _trim_user_readings(connection: sqlite3.Connection, user_id: int) -> None:
    """Keep only the newest MAX_DATASET_ROWS readings for this account."""
    connection.execute(
        """
        DELETE FROM energy_readings
        WHERE user_id = ? AND id NOT IN (
            SELECT id FROM energy_readings WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
        )
        """,
        (user_id, user_id, MAX_DATASET_ROWS),
    )


def save_energy_readings(user_id: str, data: pd.DataFrame) -> int:
    records = [
        (
            int(user_id),
            pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
            float(row["temperature"]),
            float(row["humidity"]),
            float(row["wind_speed"]),
            float(row["consumption_kwh"]),
        )
        for _, row in data.iterrows()
    ]
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO energy_readings (
                user_id, timestamp, temperature, humidity, wind_speed, consumption_kwh
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, timestamp) DO UPDATE SET
                temperature = excluded.temperature,
                humidity = excluded.humidity,
                wind_speed = excluded.wind_speed,
                consumption_kwh = excluded.consumption_kwh
            """,
            records,
        )
        _trim_user_readings(connection, int(user_id))
    return len(records)


def get_energy_readings(user_id: str) -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT id, timestamp, temperature, humidity, wind_speed, consumption_kwh, created_at
            FROM energy_readings
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            connection,
            params=(int(user_id), MAX_DATASET_ROWS),
        )


def add_energy_reading(
    user_id: str,
    timestamp: datetime,
    temperature: float,
    humidity: float,
    wind_speed: float,
    consumption_kwh: float,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO energy_readings (
                user_id, timestamp, temperature, humidity, wind_speed, consumption_kwh
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, timestamp) DO UPDATE SET
                temperature = excluded.temperature,
                humidity = excluded.humidity,
                wind_speed = excluded.wind_speed,
                consumption_kwh = excluded.consumption_kwh
            """,
            (
                int(user_id),
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                float(temperature),
                float(humidity),
                float(wind_speed),
                float(consumption_kwh),
            ),
        )
        _trim_user_readings(connection, int(user_id))


def cap_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Enforce the MAX_DATASET_ROWS limit by keeping the newest readings.
    Returns (dataset, number_of_older_rows_left_out)."""
    if len(df) <= MAX_DATASET_ROWS:
        return df, 0
    ordered = df
    if "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce")
        ordered = df.iloc[stamps.argsort(kind="stable")]
    return ordered.tail(MAX_DATASET_ROWS).reset_index(drop=True), len(df) - MAX_DATASET_ROWS


def load_dashboard_data(source: str, user_id: str, uploaded_file: object | None, location_type: str = "Town") -> pd.DataFrame:
    if source == "SQLite readings":
        saved_data = get_energy_readings(user_id)
        if not saved_data.empty:
            return saved_data[REQUIRED_DATA_COLUMNS]
        st.sidebar.warning(t("warn_no_saved"))
        return generate_sample_data(location_type)
    if source == "Upload CSV" and uploaded_file is not None:
        uploaded, dropped = cap_dataset(pd.read_csv(uploaded_file))
        if dropped:
            st.sidebar.warning(t("warn_capped", max=MAX_DATASET_ROWS, dropped=dropped))
            signature = (getattr(uploaded_file, "name", ""), dropped)
            if st.session_state.get("capped_notice") != signature:  # toast once per file, not every rerun
                st.session_state["capped_notice"] = signature
                st.toast(t("toast_capped", max=MAX_DATASET_ROWS))
        return uploaded
    return generate_sample_data(location_type)


def create_live_reading(user_id: str, data: pd.DataFrame) -> None:
    saved_data = get_energy_readings(user_id)
    source_timestamp = pd.to_datetime(data["timestamp"].max())
    last_timestamp = source_timestamp if saved_data.empty else max(
        source_timestamp, pd.to_datetime(saved_data["timestamp"].max())
    )
    next_timestamp = last_timestamp + pd.Timedelta(hours=1)
    recent_average = float(data["consumption_kwh"].tail(min(24, len(data))).mean())
    hour_wave = 0.55 * math.sin((next_timestamp.hour - 7) * math.pi / 12)
    temperature = round(float(data["temperature"].tail(24).mean()) + hour_wave, 1)
    humidity = round(max(20.0, min(95.0, float(data["humidity"].tail(24).mean()) - hour_wave * 4)), 1)
    wind_speed = round(max(0.0, float(data["wind_speed"].tail(24).mean()) + hour_wave * 0.3), 1)
    consumption = round(max(0.2, recent_average * (1.0 + hour_wave * 0.08)), 2)
    add_energy_reading(
        user_id=user_id,
        timestamp=next_timestamp.to_pydatetime(),
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
        consumption_kwh=consumption,
    )


USAGE_LEVEL_COLORS = {"Low": "#0d9488", "Medium": "#d97706", "High": "#dc2626"}
USAGE_LEVEL_BACKGROUNDS = {"Low": "#ccfbf1", "Medium": "#fef3c7", "High": "#fee2e2"}


def classify_usage_level(values: pd.Series) -> pd.Series:
    """Label each reading Low / Medium / High relative to the series' own terciles."""
    if values.empty:
        return values.astype(str)
    low_cut = values.quantile(1 / 3)
    high_cut = values.quantile(2 / 3)

    def _label(v: float) -> str:
        if v <= low_cut:
            return "Low"
        if v >= high_cut:
            return "High"
        return "Medium"

    return values.apply(_label)


def generate_energy_tips(df: pd.DataFrame, location_type: str | None) -> list[dict]:
    """A short list of practical, partly data-driven energy-saving suggestions."""
    tips: list[dict] = []
    kwh = df["consumption_kwh"].astype(float)
    ts = pd.to_datetime(df["timestamp"])

    hourly = kwh.groupby(ts.dt.hour.values).mean()
    if len(hourly) >= 6:
        peak_hour = int(hourly.idxmax())
        off_hours = sorted(hourly.nsmallest(2).index.tolist())
        off_hours_str = " & ".join(f"{h:02d}:00" for h in off_hours)
        tips.append({
            "icon": "⏰", "title": t("tip_peak_shift_title"),
            "detail": t("tip_peak_shift_detail", peak=f"{peak_hour:02d}:00", off=off_hours_str),
        })

    slope = _slope(df["temperature"], kwh)
    if slope is not None and slope > 0:
        tips.append({"icon": "🌡️", "title": t("tip_cooling_title"), "detail": t("tip_cooling_detail")})

    weekend_mask = ts.dt.weekday.values >= 5
    if weekend_mask.any() and (~weekend_mask).any():
        wk_mean = float(kwh[weekend_mask].mean())
        wd_mean = float(kwh[~weekend_mask].mean())
        if wk_mean > wd_mean * 1.1:
            tips.append({"icon": "📅", "title": t("tip_weekend_high_title"), "detail": t("tip_weekend_high_detail")})
        elif wd_mean > wk_mean * 1.1:
            tips.append({"icon": "📅", "title": t("tip_weekday_high_title"), "detail": t("tip_weekday_high_detail")})

    levels = classify_usage_level(kwh)
    high_share = float((levels == "High").mean()) * 100 if len(levels) else 0.0
    if high_share >= 40:
        tips.append({
            "icon": "⚡", "title": t("tip_high_share_title"),
            "detail": t("tip_high_share_detail", pct=high_share),
        })

    if location_type == "City":
        tips.append({"icon": "🏢", "title": t("tip_city_title"), "detail": t("tip_city_detail")})
    elif location_type == "Village":
        tips.append({"icon": "🌾", "title": t("tip_village_title"), "detail": t("tip_village_detail")})

    for key in ("led", "standby"):
        tips.append({"icon": "💡", "title": t(f"tip_{key}_title"), "detail": t(f"tip_{key}_detail")})

    return tips[:6]


def render_energy_tips(df: pd.DataFrame, location_type: str | None) -> None:
    """Card with a short list of energy-saving suggestions for the current data."""
    tips = generate_energy_tips(df, location_type)
    if not tips:
        return
    rows_html = "".join(
        '<div style="display:flex;gap:0.8rem;align-items:flex-start;background:#fffbeb;'
        'border-left:4px solid #f59e0b;border-radius:10px;padding:0.7rem 0.9rem;margin-top:0.55rem;">'
        f'<div style="font-size:1.35rem;line-height:1.2;">{tip["icon"]}</div>'
        f'<div style="flex:1;"><div style="font-weight:700;color:#1e293b;">{tip["title"]}</div>'
        f'<div style="color:#475569;font-size:0.88rem;margin-top:2px;">{tip["detail"]}</div></div></div>'
        for tip in tips
    )
    with st.container(border=True):
        st.subheader(t("tips_title"))
        st.caption(t("tips_caption"))
        st.markdown(rows_html, unsafe_allow_html=True)


ROOM_PROFILES: dict[str, list[tuple[str, float]]] = {
    # Typical household shares (sum to 1.0); City leans more cooling-heavy, Village less so.
    "Village": [("kitchen", 0.30), ("living", 0.20), ("bedroom", 0.20), ("water_heater", 0.15), ("lighting_other", 0.15)],
    "Town": [("kitchen", 0.25), ("living", 0.20), ("bedroom", 0.25), ("water_heater", 0.15), ("lighting_other", 0.15)],
    "City": [("kitchen", 0.20), ("living", 0.18), ("bedroom", 0.35), ("water_heater", 0.12), ("lighting_other", 0.15)],
}
ROOM_ICONS = {"kitchen": "🍳", "living": "📺", "bedroom": "🛏️", "water_heater": "🚿", "lighting_other": "💡"}
ROOM_COLORS = {
    "kitchen": "#f59e0b", "living": "#4f46e5", "bedroom": "#0d9488",
    "water_heater": "#dc2626", "lighting_other": "#7c3aed",
}


def render_room_usage(df: pd.DataFrame, location_type: str | None) -> None:
    """Typical room/appliance split, scaled to this dataset's actual usage.
    Not measured per room (the app has no per-room sensors) - this applies common
    household percentages to the real total so it's a reasonable illustrative guide."""
    if df.empty:
        return
    profile = ROOM_PROFILES.get(location_type or "Town", ROOM_PROFILES["Town"])
    bill_info = estimate_units_and_bill(df)
    total_units = bill_info["total_units"]
    total_bill = bill_info["bill"]

    rows_html = ""
    for key, share in profile:
        units = total_units * share
        cost = total_bill * share
        rows_html += (
            '<div style="display:flex;align-items:center;gap:0.9rem;padding:0.6rem 0;'
            f'border-bottom:1px solid rgba(15,23,42,0.07);">'
            f'<div style="font-size:1.3rem;width:34px;text-align:center;">{ROOM_ICONS[key]}</div>'
            '<div style="flex:1;">'
            f'<div style="display:flex;justify-content:space-between;font-weight:600;color:#1e293b;">'
            f'<span>{t(f"room_{key}")}</span><span>{share * 100:.0f}%</span></div>'
            f'<div style="background:rgba(15,23,42,0.08);border-radius:999px;height:7px;margin-top:5px;">'
            f'<div style="background:{ROOM_COLORS[key]};width:{share * 100:.0f}%;height:100%;border-radius:999px;"></div></div>'
            '</div>'
            f'<div style="text-align:right;min-width:92px;">'
            f'<div style="font-weight:700;color:#1e293b;">{units:.1f} kWh</div>'
            f'<div style="font-size:0.78rem;color:#64748b;">{t("room_est_cost", v=cost)}</div></div>'
            '</div>'
        )

    with st.container(border=True):
        st.subheader(t("room_usage_title"))
        st.caption(t("room_usage_caption"))
        st.markdown(rows_html, unsafe_allow_html=True)
        st.caption(t("room_usage_disclaimer"))


def usage_level_for(value: float, series: pd.Series) -> str:
    """Low / Medium / High for one value, using the same terciles as the breakdown card."""
    low_cut, high_cut = series.quantile(1 / 3), series.quantile(2 / 3)
    if value <= low_cut:
        return "Low"
    if value >= high_cut:
        return "High"
    return "Medium"


def _slope(x: pd.Series, y: pd.Series) -> float | None:
    """kWh change per unit change of x in this dataset (None if it can't be estimated)."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 10 or float(x[mask].std()) == 0.0:
        return None
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def analyze_usage_reasons(row, ref: pd.DataFrame, location_type: str | None = None) -> list[dict]:
    """Work out what is pushing a reading up or down, using the readings in `ref`."""
    ts = pd.Timestamp(row["timestamp"])
    ref_ts = pd.to_datetime(ref["timestamp"])
    kwh = ref["consumption_kwh"].astype(float)
    overall = float(kwh.mean()) or 1e-9
    reasons: list[dict] = []

    # 1) Time of day
    hourly = kwh.groupby(ref_ts.dt.hour.values).mean()
    if len(hourly) >= 6 and ts.hour in hourly.index:
        this_hour = float(hourly[ts.hour])
        impact = this_hour - overall
        if this_hour >= float(hourly.quantile(2 / 3)):
            key, effect = "peak_hour", "up"
        elif this_hour <= float(hourly.quantile(1 / 3)):
            key, effect = "offpeak_hour", "down"
        else:
            key, effect = "normal_hour", "neutral"
        reasons.append({
            "icon": "⏰", "effect": effect, "impact": impact,
            "title": t(f"reason_{key}_title"),
            "detail": t(f"reason_{key}_detail", hour=ts.strftime("%I %p").lstrip("0"),
                        v=this_hour, p=abs(impact / overall * 100)),
        })

    # 2) Temperature (only claimed when this dataset shows heat really does raise usage)
    temp = float(row["temperature"])
    avg_temp = float(pd.to_numeric(ref["temperature"], errors="coerce").mean())
    diff = temp - avg_temp
    slope = _slope(ref["temperature"], kwh)
    if slope is None or slope > 0:
        impact, est = None, ""
        if slope is not None and abs(diff) >= 2:
            impact = slope * diff
            est = t("est_effect", e=impact)
        if diff >= 2:
            key, effect = "hot", "up"
        elif diff <= -2:
            key, effect = "cool", "down"
        else:
            key, effect = "temp_normal", "neutral"
        reasons.append({
            "icon": "🌡️", "effect": effect, "impact": impact,
            "title": t(f"reason_{key}_title"),
            "detail": t(f"reason_{key}_detail", temp=temp, avg=avg_temp, d=abs(diff), est=est),
        })

    # 3) Weekday vs weekend
    weekend_mask = ref_ts.dt.weekday.values >= 5
    if weekend_mask.any() and (~weekend_mask).any():
        wk_mean = float(kwh[weekend_mask].mean())
        wd_mean = float(kwh[~weekend_mask].mean())
        is_weekend = ts.weekday() >= 5
        group_mean = wk_mean if is_weekend else wd_mean
        ratio = group_mean / overall
        effect = "up" if ratio >= 1.05 else "down" if ratio <= 0.95 else "neutral"
        key = "weekend" if is_weekend else "weekday"
        reasons.append({
            "icon": "📅", "effect": effect, "impact": group_mean - overall,
            "title": t(f"reason_{key}_title"),
            "detail": t(f"reason_{key}_detail", w=wk_mean, d=wd_mean),
        })

    # 4) Humidity and wind (qualitative only)
    humidity = float(row["humidity"])
    if humidity >= max(float(pd.to_numeric(ref["humidity"], errors="coerce").mean()) + 10, 75):
        reasons.append({"icon": "💧", "effect": "up", "impact": None,
                        "title": t("reason_humid_title"), "detail": t("reason_humid_detail", h=humidity)})
    wind = float(row["wind_speed"])
    if wind >= float(pd.to_numeric(ref["wind_speed"], errors="coerce").mean()) + 3:
        reasons.append({"icon": "🌬️", "effect": "down", "impact": None,
                        "title": t("reason_windy_title"), "detail": t("reason_windy_detail", w=wind)})

    # 5) Nothing stands out?
    if not any(r["effect"] in ("up", "down") for r in reasons):
        reasons.append({"icon": "✅", "effect": "neutral", "impact": None,
                        "title": t("reason_none_title"), "detail": t("reason_none_detail")})

    # 6) Location context
    if location_type in ("Village", "Town", "City"):
        reasons.append({
            "icon": "📍", "effect": "info", "impact": None,
            "title": t("reason_loc_title", loc=t(location_type.lower())),
            "detail": t(f"reason_loc_{location_type.lower()}"),
        })

    # Flag the strongest measurable driver (at least 5% of the average) as the main cause
    measurable = [r for r in reasons if r["effect"] in ("up", "down") and r["impact"] is not None]
    for r in reasons:
        r["main"] = False
    if measurable:
        top = max(measurable, key=lambda r: abs(r["impact"]))
        if abs(top["impact"]) >= 0.05 * overall:
            top["main"] = True

    order = {"up": 0, "down": 1, "neutral": 2, "info": 3}
    reasons.sort(key=lambda r: (not r["main"], order[r["effect"]]))
    return reasons


REASON_STYLES = {
    "up": ("#dc2626", "#fef2f2", "tag_raises"),
    "down": ("#0d9488", "#f0fdfa", "tag_lowers"),
    "neutral": ("#64748b", "#f8fafc", "tag_neutral"),
    "info": ("#4f46e5", "#eef2ff", "tag_context"),
}


def render_usage_reasons(
    title_key: str, row, ref: pd.DataFrame, value: float, location_type: str | None = None, **title_kwargs
) -> None:
    """Card explaining *why* a reading sits at its Low / Medium / High level."""
    level = usage_level_for(value, ref["consumption_kwh"])
    avg = float(ref["consumption_kwh"].mean())
    pct = (value / avg - 1) * 100 if avg else 0.0
    reasons = analyze_usage_reasons(row, ref, location_type)

    if pct >= 5:
        summary = t("reasons_summary_above", v=value, p=pct, avg=avg)
    elif pct <= -5:
        summary = t("reasons_summary_below", v=value, p=abs(pct), avg=avg)
    else:
        summary = t("reasons_summary_near", v=value, avg=avg)

    level_chip = (
        f'<span style="background:{USAGE_LEVEL_BACKGROUNDS[level]};color:{USAGE_LEVEL_COLORS[level]};'
        f'border-radius:999px;padding:3px 14px;font-weight:800;font-size:0.85rem;">'
        f'{t(level.lower())}</span>'
    )
    rows_html = ""
    for r in reasons:
        color, background, tag_key = REASON_STYLES[r["effect"]]
        tag = (
            f'<span style="background:{color};color:#fff;border-radius:999px;padding:1px 9px;'
            f'font-size:0.7rem;font-weight:700;margin-left:8px;">{t(tag_key)}</span>'
        )
        if r["main"]:
            tag += (
                '<span style="background:#f59e0b;color:#1e293b;border-radius:999px;padding:1px 9px;'
                f'font-size:0.7rem;font-weight:800;margin-left:6px;">★ {t("tag_main")}</span>'
            )
        highlight = "box-shadow:0 0 0 2px #f59e0b;" if r["main"] else ""
        rows_html += (
            f'<div style="display:flex;gap:0.8rem;align-items:flex-start;background:{background};'
            f'border-left:4px solid {color};border-radius:10px;padding:0.7rem 0.9rem;margin-top:0.55rem;{highlight}">'
            f'<div style="font-size:1.35rem;line-height:1.2;">{r["icon"]}</div>'
            f'<div style="flex:1;"><div style="font-weight:700;color:#1e293b;">{r["title"]}{tag}</div>'
            f'<div style="color:#475569;font-size:0.88rem;margin-top:2px;">{r["detail"]}</div></div></div>'
        )

    with st.container(border=True):
        st.subheader(t(title_key, level=t(level.lower()), **title_kwargs))
        st.markdown(
            f'<div style="margin-bottom:0.2rem;">{level_chip}'
            f'<span style="margin-left:10px;color:#334155;font-weight:600;">{summary}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(t("reasons_caption"))
        st.markdown(rows_html, unsafe_allow_html=True)


def estimate_units_and_bill(df: pd.DataFrame) -> dict:
    """Units (kWh) used, per-day rate, 60-day projection and the estimated TN EB two-month bill."""
    kwh = pd.to_numeric(df["consumption_kwh"], errors="coerce").dropna()
    stamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values()
    step = stamps.diff().median() if len(stamps) > 1 else pd.Timedelta(hours=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=1)
    span_days = max(((stamps.max() - stamps.min()) + step).total_seconds() / 86400, 1 / 24) if len(stamps) else 1.0
    total_units = float(kwh.sum())
    per_day = total_units / span_days
    units_60 = per_day * 60
    return {
        "total_units": total_units, "days": span_days, "per_day": per_day,
        "units_60": units_60, "bill": calculate_tn_eb_domestic_bill(units_60)[0],
    }


def render_units_summary(data: pd.DataFrame, data_source: str, location_type: str) -> None:
    """Card showing units used and the estimated bill; for sample data, all three locations side by side."""

    def tile(title: str, est: dict, selected: bool = False) -> str:
        below = est["bill"] < SAMPLE_BILL_LIMIT
        pill_bg, pill_fg, pill_text = (
            ("#dcfce7", "#15803d", "✓ " + t("units_below", limit=SAMPLE_BILL_LIMIT)) if below
            else ("#fef3c7", "#b45309", "⚠ " + t("units_above", limit=SAMPLE_BILL_LIMIT))
        )
        outline = "box-shadow:0 0 0 2px #4f46e5;" if selected else ""
        tag = (
            '<span style="background:#4f46e5;color:#fff;border-radius:999px;padding:1px 9px;'
            f'font-size:0.68rem;font-weight:700;margin-left:8px;">{t("units_selected")}</span>'
        ) if selected else ""
        return (
            f'<div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:0.9rem 1rem;{outline}">'
            f'<div style="font-weight:700;color:#1e293b;">{title}{tag}</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#4f46e5;margin-top:0.2rem;">'
            f'{t("units_two_month", u=est["units_60"])}</div>'
            f'<div style="color:#64748b;font-size:0.82rem;">{t("units_per_day", u=est["per_day"])}</div>'
            f'<div style="color:#1e293b;font-weight:600;margin-top:0.5rem;">{t("units_bill", b=est["bill"])}</div>'
            f'<div style="display:inline-block;margin-top:0.4rem;background:{pill_bg};color:{pill_fg};'
            f'border-radius:999px;padding:2px 11px;font-size:0.75rem;font-weight:700;">{pill_text}</div></div>'
        )

    with st.container(border=True):
        st.subheader(t("units_title"))
        st.caption(t("units_note"))
        if data_source == "Sample data":
            names = {"Village": t("village"), "Town": t("town"), "City": t("city")}
            for col, loc in zip(st.columns(3), ("Village", "Town", "City")):
                with col:
                    st.markdown(
                        tile(names[loc], estimate_units_and_bill(generate_sample_data(loc)), loc == location_type),
                        unsafe_allow_html=True,
                    )
        else:
            est = estimate_units_and_bill(data)
            st.markdown(tile(describe_dataset_source(data_source, location_type), est), unsafe_allow_html=True)
        current = estimate_units_and_bill(data)
        st.caption(t("units_dataset_total", u=current["total_units"], d=current["days"]))


def metric_card(
    label: str, value: str, delta: str | None = None, series: list[float] | None = None, help: str | None = None
) -> None:
    st.metric(
        label=label,
        value=value,
        delta=delta,
        chart_data=series,
        chart_type="line",
        border=True,
        help=help,
    )


def describe_dataset_source(data_source: str, location_type: str) -> str:
    """Short, human-readable label for whatever dataset is currently active."""
    location_key = {"Village": "village", "Town": "town", "City": "city"}.get(location_type, "town")
    if data_source == "Sample data":
        return f"{t(location_key)} {t('sample_data_suffix')}"
    if data_source == "SQLite readings":
        return t("your_saved_readings")
    if data_source == "Upload CSV":
        return t("your_uploaded_csv")
    return t("sample_data_suffix")


def render_view_banner(variant: str, icon: str, title: str, subtitle: str) -> None:
    """Clean section header card, tinted differently per workspace tab."""
    st.markdown(
        f"""
        <div class="view-banner view-banner-{variant}">
            <span class="view-banner-icon">{icon}</span>
            <div>
                <div class="view-banner-title">{title}</div>
                <div class="view-banner-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_welcome_dialog() -> None:
    @st.dialog(t("dlg_welcome_title"), width="large")
    def _dialog() -> None:
        user = st.session_state.get("user")
        name = user["name"] if user else t("guest_name")
        st.markdown(f"**{t('dlg_hi', name=name)}**")
        st.write(t("dlg_map_intro"))
        st.write(t("dlg_map_dashboard"))
        st.write(t("dlg_map_forecast"))
        st.write(t("dlg_map_data"))
        st.write(t("dlg_map_sqlite"))
        st.caption(t("dlg_welcome_caption"))
        if st.button(t("got_it"), type="primary", width="stretch"):
            st.session_state["show_welcome"] = False
            st.rerun()

    _dialog()


def show_reading_guide_dialog() -> None:
    @st.dialog(t("dlg_guide_title"))
    def _dialog() -> None:
        st.write(t("dlg_guide_consumption"))
        st.write(t("dlg_guide_average"))
        st.write(t("dlg_guide_weather"))
        st.write(t("dlg_guide_status"))
        if st.button(t("close"), width="stretch", key="close_reading_guide"):
            st.rerun()

    _dialog()


def show_forecast_settings_dialog() -> None:
    @st.dialog(t("dlg_fs_title"))
    def _dialog() -> None:
        st.write(t("dlg_fs_hours"))
        st.write(t("dlg_fs_weather"))
        st.caption(t("dlg_fs_caption"))
        if st.button(t("close"), width="stretch", key="close_forecast_settings"):
            st.rerun()

    _dialog()


def show_tariff_dialog() -> None:
    @st.dialog(t("dlg_tariff_title"), width="large")
    def _dialog() -> None:
        le, gt = t("tariff_total_le_500"), t("tariff_total_gt_500")
        free = t("free")

        def rate(p: str) -> str:
            return t("rs_per_unit", p=p)

        def rng(a: int, b: int) -> str:
            return t("slab_range", a=a, b=b)

        tariff_rows = pd.DataFrame(
            [
                [le, t("slab_first", n=200), free],
                [le, rng(201, 400), rate("4.70")],
                [le, rng(401, 500), rate("6.30")],
                [gt, t("slab_first", n=100), free],
                [gt, rng(101, 400), rate("4.70")],
                [gt, rng(401, 500), rate("6.30")],
                [gt, rng(501, 600), rate("8.40")],
                [gt, rng(601, 800), rate("9.45")],
                [gt, rng(801, 1000), rate("10.50")],
                [gt, t("slab_above", n=1000), rate("11.55")],
            ],
            columns=[t("tariff_col_total"), t("tariff_col_slab"), t("tariff_col_charge")],
        )
        st.dataframe(tariff_rows, hide_index=True)
        st.caption(t("dlg_tariff_caption"))
        if st.button(t("close"), width="stretch", key="close_tariff"):
            st.rerun()

    _dialog()


def show_sqlite_info_dialog() -> None:
    @st.dialog(t("dlg_sqlite_title"))
    def _dialog() -> None:
        st.write(t("dlg_sqlite_intro"))
        st.write(t("dlg_sqlite_users"))
        st.write(t("dlg_sqlite_readings"))
        st.write(t("dlg_sqlite_limit", max=MAX_DATASET_ROWS))
        st.caption(t("dlg_sqlite_caption"))
        if st.button(t("close"), width="stretch", key="close_sqlite_info"):
            st.rerun()

    _dialog()


def show_registration_dialog(name: str, email: str) -> None:
    @st.dialog(t("dlg_reg_title"))
    def _dialog() -> None:
        st.markdown(
            '<div style="text-align:center;padding:0.4rem 0 0.2rem;">'
            '<div style="font-size:2.6rem;width:84px;height:84px;line-height:84px;margin:0 auto 0.7rem;'
            'border-radius:50%;background:#e0e7ff;">🎉</div>'
            f'<div style="font-size:1.25rem;font-weight:800;color:#1e293b;">'
            f'{escape(t("dlg_reg_welcome", name=name))}</div>'
            f'<div style="color:#475569;margin-top:0.25rem;">{escape(t("dlg_reg_body", email=email))}</div></div>',
            unsafe_allow_html=True,
        )
        st.info(t("dlg_reg_next", login=t("login_tab")))
        st.caption(t("dlg_reg_privacy", max=MAX_DATASET_ROWS))
        if st.button(t("got_it"), type="primary", width="stretch", key="close_registration"):
            st.rerun()

    _dialog()


def add_auth_page_styles() -> None:
    """Scoped styling for the login/register screen only: a clean split layout —
    solid indigo brand panel on the left, themed form card on the right."""
    p = THEME_PALETTES.get(st.session_state.get("theme", "Light"), THEME_PALETTES["Light"])
    st.html(
        f"""
        <style>
        .stApp {{ background: {p['app_bg']} !important; }}
        .stApp [data-testid="stHeader"] {{ background: transparent !important; }}

        @keyframes auth-card-enter {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes auth-feature-enter {{
            from {{ opacity: 0; transform: translateX(-10px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}

        .auth-hero {{
            position: relative; padding: 2.25rem 2rem; border-radius: 20px; overflow: hidden;
            background: linear-gradient(155deg, #4338ca 0%, #4f46e5 55%, #6366f1 100%);
            box-shadow: 0 1px 2px rgba(67,56,202,0.3), 0 16px 40px rgba(67,56,202,0.25);
            animation: auth-card-enter 420ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        .auth-hero-content {{ position: relative; z-index: 1; }}
        .auth-logo {{
            font-size: 1.9rem; width: 56px; height: 56px; display: flex; align-items: center; justify-content: center;
            background: rgba(255,255,255,0.16); border-radius: 14px; margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.25);
        }}
        .auth-title {{ font-size: clamp(1.7rem, 3.4vw, 2.25rem); font-weight: 800; margin: 0 0 0.5rem 0; line-height: 1.15; color: #ffffff !important; }}
        .auth-subtitle {{ color: rgba(255,255,255,0.85) !important; font-size: 0.98rem; max-width: 26rem; margin-bottom: 1.6rem; line-height: 1.6; }}
        .auth-feature {{
            display: flex; align-items: center; gap: 0.85rem; margin-bottom: 1rem;
            animation: auth-feature-enter 400ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        .auth-feature:nth-child(4) {{ animation-delay: 60ms; }}
        .auth-feature:nth-child(5) {{ animation-delay: 120ms; }}
        .auth-feature:nth-child(6) {{ animation-delay: 180ms; }}
        .auth-feature-icon {{
            flex: 0 0 auto; width: 38px; height: 38px; border-radius: 10px; display: flex;
            align-items: center; justify-content: center; font-size: 1.1rem;
            background: rgba(255,255,255,0.16) !important; color: #ffffff !important;
        }}
        .auth-feature strong {{ color: #ffffff !important; font-size: 0.94rem; }}
        .auth-feature span {{ color: rgba(255,255,255,0.75) !important; font-size: 0.84rem; }}

        [class*="st-key-auth_card"] {{
            background: {p['card_bg']}; border: 1px solid {p['card_border']}; border-radius: 20px;
            padding: 1.75rem 1.5rem 1.25rem 1.5rem; box-shadow: 0 1px 2px {p['card_shadow']}, 0 16px 40px {p['card_shadow']};
            animation: auth-card-enter 420ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }}
        [class*="st-key-auth_card"] h3, [class*="st-key-auth_card"] p {{ color: {p['text_primary']} !important; }}

        [class*="st-key-auth_card"] [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            gap: 1.25rem; background: transparent; border-bottom: 1px solid {p['tab_border']}; border-radius: 0;
        }}
        [class*="st-key-auth_card"] [data-testid="stTabs"] button[data-baseweb="tab"] {{
            color: {p['tab_text']} !important; font-weight: 600 !important; transition: color 180ms ease !important;
        }}
        [class*="st-key-auth_card"] [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
            color: {p['accent']} !important;
        }}
        [class*="st-key-auth_card"] [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            background-color: {p['accent']} !important; height: 2px !important;
        }}
        [class*="st-key-auth_card"] [data-testid="stTabs"] [data-testid="stTabContent"] {{ padding-top: 1.1rem; }}

        [class*="st-key-auth_card"] label,
        [class*="st-key-auth_card"] [data-testid="stWidgetLabel"] p {{ color: {p['text_secondary']} !important; font-weight: 600 !important; }}
        [class*="st-key-auth_card"] input {{
            background: {p['input_bg']} !important; border: 1px solid {p['input_border']} !important;
            border-radius: 10px !important; color: {p['input_text']} !important; caret-color: {p['input_text']} !important;
            transition: box-shadow 180ms ease, border-color 180ms ease !important;
        }}
        [class*="st-key-auth_card"] input:focus {{
            border-color: {p['accent']} !important; box-shadow: 0 0 0 3px {p['accent_soft']} !important;
        }}
        [class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] button {{
            background: {p['accent']} !important; border: none !important;
            color: #ffffff !important; border-radius: 10px !important; font-weight: 600 !important;
            box-shadow: 0 1px 2px {p['accent_soft']}, 0 6px 16px {p['accent_soft']} !important;
            transition: transform 150ms ease, box-shadow 150ms ease, background-color 150ms ease !important;
        }}
        [class*="st-key-auth_card"] [data-testid="stFormSubmitButton"] button:hover {{
            background: {p['accent_hover']} !important; transform: translateY(-1px) !important;
        }}
        @media (prefers-reduced-motion: reduce) {{
            .auth-hero, .auth-feature, [class*="st-key-auth_card"] {{ animation: none !important; }}
        }}
        </style>
        """
    )


def show_auth_page() -> None:
    add_auth_page_styles()

    hero_col, form_col = st.columns([5, 4], gap="large")

    with hero_col:
        st.markdown(
            f"""
            <div class="auth-hero">
                <div class="auth-hero-content">
                    <div class="auth-logo">⚡</div>
                    <div class="auth-title">{t('app_title')}</div>
                    <div class="auth-subtitle">{t('app_subtitle')}</div>
                    <div class="auth-feature">
                        <span class="auth-feature-icon">📈</span>
                        <div><strong>{t('feature_forecast_title')}</strong><br><span>{t('feature_forecast_body')}</span></div>
                    </div>
                    <div class="auth-feature">
                        <span class="auth-feature-icon">🌦️</span>
                        <div><strong>{t('feature_weather_title')}</strong><br><span>{t('feature_weather_body')}</span></div>
                    </div>
                    <div class="auth-feature">
                        <span class="auth-feature-icon">🔒</span>
                        <div><strong>{t('feature_private_title')}</strong><br><span>{t('feature_private_body')}</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        with st.container(key="auth_card"):
            login_tab, register_tab = st.tabs(
                [f":material/login: {t('login_tab')}", f":material/person_add: {t('register_tab')}"]
            )
            with login_tab:
                st.markdown(f"###### {t('welcome_back')}")
                st.caption(t("welcome_back_sub"))
                with st.form("login_form"):
                    email = st.text_input(f":material/mail: {t('email')}")
                    password = st.text_input(f":material/lock: {t('password')}", type="password")
                    submitted = st.form_submit_button(t("login_button"), type="primary", width="stretch")
                if submitted:
                    user = verify_user(email, password)
                    if user is None:
                        st.error(t("err_invalid_login"))
                    else:
                        st.session_state["user"] = user
                        st.rerun()
            with register_tab:
                st.markdown(f"###### {t('create_account_heading')}")
                st.caption(t("create_account_sub"))
                with st.form("register_form"):
                    name = st.text_input(f":material/badge: {t('name')}")
                    email = st.text_input(f":material/mail: {t('email')}", key="register_email")
                    password = st.text_input(f":material/lock: {t('password')}", type="password", key="register_password")
                    confirm_password = st.text_input(f":material/lock_reset: {t('confirm_password')}", type="password")
                    submitted = st.form_submit_button(t("create_account_button"), type="primary", width="stretch")
                if submitted:
                    if password != confirm_password:
                        st.error(t("err_pw_mismatch"))
                    else:
                        success, message = create_user(name, email, password)
                        if success:
                            st.success(t("reg_success_line"))
                            show_registration_dialog(name.strip() or email, email)
                        else:
                            st.error(message)


def render_live_status(user_id: str, run_every: str | None) -> None:
    @st.fragment(run_every=run_every)
    def live_status() -> None:
        readings = get_energy_readings(user_id)
        if readings.empty:
            st.info(t("live_none"))
            return
        latest = readings.iloc[0]
        st.caption(
            t("live_latest", ts=pd.to_datetime(latest['timestamp']).strftime('%d %b %Y, %I:%M %p'))
        )
        st.metric(t("live_consumption"), f"{latest['consumption_kwh']:.2f} kWh", border=True)
    live_status()


init_db()
init_energy_table()
st.session_state.setdefault("user", None)

st.session_state.setdefault("lang", "English")


def set_language(language: str) -> None:
    """Button callback: runs before the rerun, so every t() call sees the new language."""
    st.session_state["lang"] = language


def set_theme(theme: str) -> None:
    """Button callback: runs before the rerun, so add_transition_styles() picks it up."""
    st.session_state["theme"] = theme


_, toggle_col = st.columns([5, 3.6])
with toggle_col:
    theme_col, lang_col = st.columns([1.3, 2.4])
    with theme_col:
        light_col, dark_col = st.columns(2)
        with light_col:
            st.button(
                "☀️", key="theme_light", on_click=set_theme, args=("Light",), width="stretch",
                type="primary" if st.session_state["theme"] == "Light" else "secondary",
                help=t("theme_light"),
            )
        with dark_col:
            st.button(
                "🌙", key="theme_dark", on_click=set_theme, args=("Dark",), width="stretch",
                type="primary" if st.session_state["theme"] == "Dark" else "secondary",
                help=t("theme_dark"),
            )
    with lang_col:
        en_col, ta_col = st.columns(2)
        with en_col:
            st.button(
                "English", key="lang_en", on_click=set_language, args=("English",), width="stretch",
                type="primary" if st.session_state["lang"] == "English" else "secondary",
            )
        with ta_col:
            st.button(
                "தமிழ்", key="lang_ta", on_click=set_language, args=("தமிழ்",), width="stretch",
                type="primary" if st.session_state["lang"] == "தமிழ்" else "secondary",
            )

if st.session_state["user"] is None:
    show_auth_page()
    st.stop()

user = st.session_state["user"]

if st.session_state.setdefault("show_welcome", True):
    show_welcome_dialog()

with st.sidebar:
    st.success(f"{t('signed_in_as')} {user['name']}")
    if st.button(f":material/logout: {t('logout')}", width="stretch"):
        st.session_state["user"] = None
        st.rerun()
    st.divider()
    data_source = st.segmented_control(
        t("data_source"),
        ["Sample data", "SQLite readings", "Upload CSV"],
        default="Sample data",
        required=True,
        key="data_source",
        width="stretch",
        format_func=lambda opt: {
            "Sample data": t("sample_data"), "SQLite readings": t("sqlite_readings"), "Upload CSV": t("upload_csv"),
        }[opt],
    )
    location_type = "Town"
    uploaded_file = None
    if data_source == "Sample data":
        location_type = st.segmented_control(
            t("location_type"),
            ["Village", "Town", "City"],
            default="Town",
            required=True,
            key="location_type",
            width="stretch",
            format_func=lambda opt: {"Village": t("village"), "Town": t("town"), "City": t("city")}[opt],
            help=t("location_help"),
        )
        sample_estimate = estimate_units_and_bill(generate_sample_data(location_type))
        st.caption(t("units_sidebar", u=sample_estimate["units_60"], b=sample_estimate["bill"]))
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            t("upload_csv_label"), type=["csv"], key="dataset_upload", help=t("upload_limit_hint", max=MAX_DATASET_ROWS)
        )

dataset_label = describe_dataset_source(data_source, location_type)

with st.sidebar:
    st.caption(f":material/database: {t('showing')} **{dataset_label}**")

data = load_dashboard_data(data_source, user["id"], uploaded_file, location_type)
try:
    validate_dataset(data)
    data = add_time_features(data)
except ValueError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.divider()
    settings_title_col, settings_info_col = st.columns([5, 1])
    with settings_title_col:
        st.subheader(t("forecast_settings"))
    with settings_info_col:
        st.markdown('<div class="info-btn">', unsafe_allow_html=True)
        if st.button(":material/info:", key="forecast_settings_info", help=t("fs_info_help")):
            show_forecast_settings_dialog()
        st.markdown("</div>", unsafe_allow_html=True)
    forecast_hours = st.slider(t("forecast_hours"), min_value=6, max_value=168, value=48, step=6)
    future_temperature = st.number_input(t("exp_temp"), value=float(data["temperature"].tail(24).mean()))
    future_humidity = st.number_input(t("exp_humidity"), value=float(data["humidity"].tail(24).mean()))
    future_wind_speed = st.number_input(t("exp_wind"), value=float(data["wind_speed"].tail(24).mean()))
    st.divider()
    st.subheader(t("live_readings"))
    if st.button(f":material/bolt: {t('gen_live')}", type="primary", width="stretch"):
        create_live_reading(user["id"], data)
        st.toast(t("toast_live_saved"))
        st.rerun()
    if st.button(f":material/database: {t('save_current')}", width="stretch"):
        saved_count = save_energy_readings(user["id"], data)
        st.toast(t("toast_saved_n", n=saved_count))
    auto_refresh = st.toggle(t("refresh_live"), value=False)

st.title("EnergySmart")
st.caption(t("page_tagline"))
st.markdown(f":green-badge[{t('badge_live')}] :blue-badge[{t('badge_ready')}]")
model, evaluation, metrics = train_cached_model(data)
recent_values = data["consumption_kwh"].tail(12).astype(float).tolist()
latest_usage = float(data["consumption_kwh"].iloc[-1])
previous_usage = float(data["consumption_kwh"].iloc[-2]) if len(data) > 1 else latest_usage

with st.container(horizontal=True):
    metric_card(t("records"), f"{len(data):,}", series=recent_values)
    metric_card(t("average_usage"), f"{data['consumption_kwh'].mean():.2f} kWh", series=recent_values, help=t("units_help"))
    metric_card(t("latest_usage"), f"{latest_usage:.2f} kWh", f"{latest_usage - previous_usage:+.2f} kWh", recent_values)
    metric_card(
        t("model_fit"), f"{metrics['r2']:.2f}",
        help=t("r2_help"),
    )

view = st.segmented_control(
    t("workspace"),
    ["Dashboard", "Forecast", "Data", "SQLite database"],
    default="Dashboard",
    required=True,
    key="workspace_view",
    format_func=lambda opt: {
        "Dashboard": t("dashboard"), "Forecast": t("forecast"), "Data": t("data"), "SQLite database": t("sqlite_database"),
    }[opt],
)

view_container = st.container(key=f"view-{view.replace(' ', '-').lower()}")
with view_container:
    if view == "Dashboard":
        render_view_banner(
            "dashboard", "📊", t("dashboard"),
            t("dashboard_subtitle", label=dataset_label),
        )
        start_date = data["timestamp"].min().date()
        end_date = data["timestamp"].max().date()
        selected_range = st.date_input(t("date_range"), value=(start_date, end_date), min_value=start_date, max_value=end_date)
        filtered_data = data
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            start, end = selected_range
            filtered_data = data[data["timestamp"].dt.date.between(start, end)]

        latest_reading = filtered_data.iloc[-1]
        average_usage = float(filtered_data["consumption_kwh"].mean())
        difference_from_average = float(latest_reading["consumption_kwh"] - average_usage)
        if latest_reading["consumption_kwh"] >= average_usage * 1.15:
            reading_status = t("status_higher")
            reading_message = t("msg_higher")
        elif latest_reading["consumption_kwh"] <= average_usage * 0.85:
            reading_status = t("status_lower")
            reading_message = t("msg_lower")
        else:
            reading_status = t("status_close")
            reading_message = t("msg_close")

        with st.container(border=True):
            st.subheader(t("latest_reading_explained"))
            st.caption(latest_reading["timestamp"].strftime("%d %b %Y, %I:%M %p"))
            reading_cols = st.columns(3)
            with reading_cols[0]:
                st.metric(
                    t("electricity_used"),
                    f"{latest_reading['consumption_kwh']:.2f} kWh",
                    t("from_average", d=difference_from_average),
                )
            with reading_cols[1]:
                st.metric(t("temperature"), f"{latest_reading['temperature']:.1f} C")
                st.caption(t("humidity_line", h=latest_reading['humidity']))
            with reading_cols[2]:
                st.badge(reading_status, icon=":material/insights:", color="blue")
                st.write(reading_message)

        render_usage_reasons(
            "reasons_title", latest_reading, filtered_data, float(latest_reading["consumption_kwh"]),
            location_type if data_source == "Sample data" else None,
        )

        if st.button(f":material/help: {t('reading_guide_btn')}", key="reading_guide_btn"):
            show_reading_guide_dialog()

        render_room_usage(filtered_data, location_type if data_source == "Sample data" else None)
        render_energy_tips(filtered_data, location_type if data_source == "Sample data" else None)

        with st.container(border=True):
            st.subheader(t("usage_trend"))
            st.line_chart(filtered_data, x="timestamp", y="consumption_kwh", color="#14B8A6")
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.subheader(t("avg_by_hour"))
                hourly = filtered_data.groupby("hour", as_index=False)["consumption_kwh"].mean()
                st.bar_chart(hourly, x="hour", y="consumption_kwh", color="#F59E0B")
        with right:
            with st.container(border=True):
                st.subheader(t("temp_and_consumption"))
                st.scatter_chart(filtered_data, x="temperature", y="consumption_kwh", color="humidity", size="wind_speed")
        with st.container(border=True):
            st.subheader(t("live_sqlite_status"))
            render_live_status(user["id"], "20s" if auto_refresh else None)

    elif view == "Forecast":
        render_view_banner(
            "forecast", "🔮", t("forecast"),
            t("forecast_subtitle", label=dataset_label),
        )
        future = build_future_frame(
            last_timestamp=pd.to_datetime(data["timestamp"].max()),
            periods=forecast_hours,
            frequency="h",
            temperature=future_temperature,
            humidity=future_humidity,
            wind_speed=future_wind_speed,
        )
        future["forecast_kwh"] = model.predict(future[FEATURE_COLUMNS])

        forecast_total = float(future["forecast_kwh"].sum())
        forecast_average = float(future["forecast_kwh"].mean())
        historical_average = float(data["consumption_kwh"].mean())
        peak_row = future.loc[future["forecast_kwh"].idxmax()]
        low_row = future.loc[future["forecast_kwh"].idxmin()]
        usage_change = (forecast_average - historical_average) / historical_average * 100
        high_threshold = forecast_average * 1.15
        low_threshold = forecast_average * 0.85
        future["usage_level"] = "Normal"
        future.loc[future["forecast_kwh"] >= high_threshold, "usage_level"] = "High"
        future.loc[future["forecast_kwh"] <= low_threshold, "usage_level"] = "Low"

        st.subheader(t("forecast_at_glance"))
        projected_bimonthly_units = round(forecast_average * 24 * 60, 1)
        bimonthly_units = st.number_input(
            t("exp_units_label"),
            min_value=0.0,
            value=projected_bimonthly_units,
            step=10.0,
            help=t("exp_units_help"),
        )
        tn_eb_bill, free_units = calculate_tn_eb_domestic_bill(bimonthly_units)
        forecast_cards = st.columns(4)
        with forecast_cards[0]:
            st.metric(t("expected_energy"), f"{forecast_total:.1f} kWh", border=True)
        with forecast_cards[1]:
            st.metric(t("tn_eb_estimate"), f"Rs. {tn_eb_bill:.0f}", border=True)
        with forecast_cards[2]:
            st.metric(
                t("peak_demand"),
                f"{peak_row['forecast_kwh']:.2f} kWh",
                peak_row["timestamp"].strftime("%d %b, %I %p"),
                border=True,
            )
        with forecast_cards[3]:
            st.metric(
                t("compared_usual"),
                f"{usage_change:+.1f}%",
                border=True,
            )

        if bimonthly_units <= 500:
            st.success(t("free_scheme_within", n=free_units))
        else:
            st.info(t("free_scheme_above", n=free_units))
        st.caption(t("dlg_tariff_caption"))
        if st.button(f":material/info: {t('view_tariff_btn')}", key="tariff_info_btn"):
            show_tariff_dialog()

        if usage_change >= 10:
            st.warning(t("demand_high", time=peak_row['timestamp'].strftime('%I:%M %p')))
        elif usage_change <= -10:
            st.success(t("demand_low"))
        else:
            st.info(t("demand_normal", v=low_row['forecast_kwh'], time=low_row['timestamp'].strftime('%I:%M %p')))

        render_usage_reasons(
            "reasons_forecast_title",
            {
                "timestamp": peak_row["timestamp"], "temperature": future_temperature,
                "humidity": future_humidity, "wind_speed": future_wind_speed,
            },
            data, float(peak_row["forecast_kwh"]),
            location_type if data_source == "Sample data" else None,
            time=peak_row["timestamp"].strftime("%I:%M %p"),
        )

        with st.container(border=True):
            st.subheader(t("actual_vs_pred"))
            st.line_chart(evaluation, x="timestamp", y=["consumption_kwh", "predicted_kwh"])
        score_cols = st.columns(3)
        with score_cols[0]:
            metric_card(
                t("mae"), f"{metrics['mae']:.2f} kWh",
                help=t("mae_help"),
            )
        with score_cols[1]:
            metric_card(
                t("rmse"), f"{metrics['rmse']:.2f} kWh",
                help=t("rmse_help"),
            )
        with score_cols[2]:
            metric_card(
                t("r2_score"), f"{metrics['r2']:.2f}",
                help=t("r2_help"),
            )
        with st.container(border=True):
            st.subheader(t("future_forecast"))
            st.area_chart(future, x="timestamp", y="forecast_kwh", color="#2563EB")
        st.subheader(t("hourly_schedule"))
        schedule = future[["timestamp", "forecast_kwh", "usage_level"]].copy()
        schedule["usage_level"] = schedule["usage_level"].map(
            {"Low": t("low"), "Normal": t("normal"), "High": t("high")}
        )
        st.dataframe(
            schedule,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(t("col_time"), format="DD MMM, hh:mm a"),
                "forecast_kwh": st.column_config.NumberColumn(t("col_expected"), format="%.2f kWh"),
                "usage_level": st.column_config.TextColumn(t("usage_level")),
            },
            hide_index=True,
            height=300,
        )
        st.download_button(
            f":material/download: {t('download_forecast')}",
            data=future[["timestamp", "forecast_kwh"]].to_csv(index=False),
            file_name="energysmart_forecast.csv",
            mime="text/csv",
        )

    elif view == "Data":
        render_view_banner(
            "data", "🗂️", t("dataset_preview"),
            t("data_subtitle", label=dataset_label, count=len(data)),
        )
        render_units_summary(data, data_source, location_type)
        render_room_usage(data, location_type if data_source == "Sample data" else None)

        preview_data = data.copy()
        preview_data["usage_level"] = classify_usage_level(preview_data["consumption_kwh"]).map(
            {"Low": t("low"), "Medium": t("medium"), "High": t("high")}
        )
        label_to_level = {t("low"): "Low", t("medium"): "Medium", t("high"): "High"}

        def _usage_level_style(value: str) -> str:
            level = label_to_level.get(value, value)
            color = USAGE_LEVEL_COLORS.get(level, "#334155")
            background = USAGE_LEVEL_BACKGROUNDS.get(level, "#f1f5f9")
            return f"background-color: {background}; color: {color}; font-weight: 600;"

        styled_preview = preview_data.style
        style_fn = getattr(styled_preview, "map", None) or styled_preview.applymap
        styled_preview = style_fn(_usage_level_style, subset=["usage_level"])

        st.dataframe(
            styled_preview,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(t("col_timestamp"), format="DD MMM YYYY, hh:mm a"),
                "temperature": st.column_config.NumberColumn(t("col_temperature"), format="%.1f C"),
                "humidity": st.column_config.NumberColumn(t("col_humidity"), format="%.1f %%"),
                "wind_speed": st.column_config.NumberColumn(t("col_wind"), format="%.1f km/h"),
                "consumption_kwh": st.column_config.NumberColumn(t("col_consumption"), format="%.2f kWh"),
                "usage_level": st.column_config.TextColumn(t("usage_level")),
                "hour": None, "day": None, "month": None, "weekday": None,
                "is_weekend": None, "season": None,
            },
            hide_index=True,
        )

    else:
        render_view_banner(
            "sqlite", "🗄️", t("sqlite_database"),
            t("sqlite_subtitle"),
        )
        sqlite_title_col, sqlite_info_col = st.columns([5, 1])
        with sqlite_title_col:
            st.subheader(t("saved_records"))
        with sqlite_info_col:
            st.markdown('<div class="info-btn">', unsafe_allow_html=True)
            if st.button(":material/info:", key="sqlite_info_btn", help=t("sqlite_info_help")):
                show_sqlite_info_dialog()
            st.markdown("</div>", unsafe_allow_html=True)
        with st.form("add_reading_form", border=True):
            form_cols = st.columns(3)
            with form_cols[0]:
                reading_date = st.date_input(t("f_date"))
                temperature = st.number_input(t("temperature"), value=28.0)
            with form_cols[1]:
                reading_time = st.time_input(t("f_time"))
                humidity = st.number_input(t("col_humidity"), value=60.0, min_value=0.0, max_value=100.0)
            with form_cols[2]:
                consumption_kwh = st.number_input(t("f_consumption"), value=4.5, min_value=0.0)
                wind_speed = st.number_input(t("col_wind"), value=5.0, min_value=0.0)
            submitted_reading = st.form_submit_button(t("save_reading"), type="primary")
        if submitted_reading:
            add_energy_reading(
                user_id=user["id"],
                timestamp=datetime.combine(reading_date, reading_time),
                temperature=temperature,
                humidity=humidity,
                wind_speed=wind_speed,
                consumption_kwh=consumption_kwh,
            )
            st.success(t("reading_saved"))
        saved_readings = get_energy_readings(user["id"])
        if saved_readings.empty:
            st.info(t("no_readings"))
        else:
            st.dataframe(
                saved_readings,
                column_config={
                    "id": None,
                    "timestamp": st.column_config.DatetimeColumn(t("col_timestamp"), format="DD MMM YYYY, hh:mm a"),
                    "temperature": st.column_config.NumberColumn(t("col_temperature"), format="%.1f C"),
                    "humidity": st.column_config.NumberColumn(t("col_humidity"), format="%.1f %%"),
                    "wind_speed": st.column_config.NumberColumn(t("col_wind"), format="%.1f km/h"),
                    "consumption_kwh": st.column_config.NumberColumn(t("col_consumption"), format="%.2f kWh"),
                    "created_at": st.column_config.DatetimeColumn(t("col_saved"), format="DD MMM YYYY, hh:mm a"),
                },
                hide_index=True,
            )
            st.download_button(
                f":material/download: {t('download_saved')}",
                data=saved_readings.to_csv(index=False),
                file_name="sqlite_energy_readings.csv",
                mime="text/csv",
            )
