"""Charging rules used to create the demo dataset.

These rules are a simple simulation, not measurements from real vehicles.
The same function is used to show a "simulator reference" next to the
model predictions in the web page.
"""
import numpy as np
import pandas as pd

CHARGERS = [3.3, 7.4, 11, 22, 30, 60, 120]          # kW
CHARGER_WEIGHTS = [0.08, 0.22, 0.15, 0.10, 0.15, 0.18, 0.12]
MAX_MINUTES = 720                                      # sessions over 12 h are left out


def sim_time(battery_kwh, start_soc, target_soc, charger_kw, temp_c):
    """Noise-free charging time in minutes."""
    dc = charger_kw > 22
    if dc:
        accept = min(charger_kw, min(150.0, battery_kwh * 1.3))
    else:
        accept = min(charger_kw, 7.4 if battery_kwh < 50 else 11.0)
    efficiency = 0.95 if dc else 0.90

    factor = 1.0
    if temp_c < 10:
        factor -= min(0.35, (10 - temp_c) * 0.012)
    if temp_c > 35:
        factor -= min(0.20, (temp_c - 35) * 0.015)

    power = accept * factor * efficiency
    energy_all = battery_kwh * (target_soc - start_soc) / 100
    energy_low = battery_kwh * max(0, min(target_soc, 80) - start_soc) / 100
    energy_high = energy_all - energy_low
    taper = 0.45 if dc else 0.85          # charging slows above 80 %
    hours = energy_low / power + energy_high / (power * taper)
    return hours * 60 + 3                 # 3 min for plugging in and handshake


def generate_sessions(n=1500, seed=2026):
    """Create n simulated charging sessions as a DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        battery = int(round(25 + rng.random() * 75))
        start = int(round(rng.random() * 90))
        target = int(round(start + 8 + rng.random() * (100 - start - 8)))
        charger = float(rng.choice(CHARGERS, p=CHARGER_WEIGHTS))
        temp = int(np.clip(round(26 + 9 * rng.standard_normal()), -5, 45))

        base = sim_time(battery, start, target, charger, temp)
        minutes = base * (1 + 0.05 * rng.standard_normal()) + 1.5 * rng.standard_normal()
        if minutes < 3 or minutes > MAX_MINUTES:
            continue
        rows.append((battery, start, target, charger, temp, round(minutes, 1)))

    return pd.DataFrame(
        rows,
        columns=["battery_kwh", "start_soc", "target_soc", "charger_kw", "temp_c", "charging_time_min"],
    )
