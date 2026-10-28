"""Generate a SYNTHETIC sample dataset of road accident records in India.

!! These records are made up for practice. They imitate broad patterns
!! (evening peaks, two-wheelers most involved, over-speeding as top cause,
!! a lockdown dip in 2020). Do NOT report them as real findings.
!! Replace data/accidents.csv with real data (see README) for your final project.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 24000

STATES = {
    "Uttar Pradesh": 1.6, "Tamil Nadu": 1.5, "Maharashtra": 1.3, "Madhya Pradesh": 1.3,
    "Karnataka": 1.2, "Rajasthan": 1.0, "Gujarat": 0.9, "Telangana": 0.9, "Kerala": 0.9,
    "Andhra Pradesh": 0.9, "West Bengal": 0.7, "Bihar": 0.7, "Delhi": 0.6, "Punjab": 0.5,
}
NORTH = {"Uttar Pradesh", "Rajasthan", "Madhya Pradesh", "Bihar", "Delhi", "Punjab"}
CAUSES = ["Over-speeding", "Wrong-side driving", "Drunk driving", "Distracted driving",
          "Poor visibility", "Vehicle defect", "Other"]
VEHICLES = ["Two-wheeler", "Car", "Truck", "Bus", "Auto-rickshaw", "Pedestrian involved"]
ROADS = ["National Highway", "State Highway", "Urban Road", "Rural Road"]

# Dates: lockdown period (25 Mar - 31 May 2020) gets far fewer accidents
days = pd.date_range("2018-01-01", "2022-12-31", freq="D")
w = np.ones(len(days))
w[(days >= "2020-03-25") & (days <= "2020-05-31")] = 0.2
date = rng.choice(days, size=N, p=w / w.sum())
date = pd.to_datetime(date)

# Hour of day: morning and evening peaks
hour_w = np.array([2, 1.5, 1.5, 1.5, 2, 3, 4, 6, 7, 6, 5, 5, 5, 5, 5, 5.5, 6.5, 8, 9, 9, 8, 6, 4, 3], float)
hour = rng.choice(24, size=N, p=hour_w / hour_w.sum())

state = rng.choice(list(STATES), size=N, p=np.array(list(STATES.values())) / sum(STATES.values()))
road = rng.choice(ROADS, size=N, p=[0.32, 0.20, 0.28, 0.20])
vehicle = rng.choice(VEHICLES, size=N, p=[0.42, 0.20, 0.14, 0.06, 0.08, 0.10])

month = date.month.values
is_north = np.isin(state, list(NORTH))
weather = np.full(N, "Clear", dtype=object)
weather[(np.isin(month, [6, 7, 8, 9])) & (rng.random(N) < 0.30)] = "Rain"
weather[(np.isin(month, [12, 1, 2])) & is_north & (rng.random(N) < 0.25)] = "Fog"

# Cause depends on time of day and weather
base = np.array([0.38, 0.12, 0.09, 0.14, 0.08, 0.07, 0.12])
P = np.tile(base, (N, 1))
night = (hour >= 21) | (hour <= 4)
P[night, 2] *= 2.2                          # drunk driving at night
P[weather == "Fog", 4] *= 3.5               # poor visibility in fog
P[weather == "Rain", 4] *= 1.8
P /= P.sum(axis=1, keepdims=True)
cause = np.array(CAUSES)[(P.cumsum(axis=1) < rng.random(N)[:, None]).sum(axis=1)]

# Severity
lam = np.full(N, 0.30)
lam *= np.select([road == "National Highway", road == "State Highway", road == "Urban Road"], [1.8, 1.5, 0.7], 1.0)
lam *= np.select([vehicle == "Truck", vehicle == "Bus", vehicle == "Pedestrian involved", vehicle == "Two-wheeler"],
                 [1.5, 1.4, 1.3, 1.1], 0.9)
lam *= np.where(cause == "Over-speeding", 1.5, 1.0)
lam *= np.where(night, 1.3, 1.0)
killed = rng.poisson(lam)
injured = rng.poisson(1.4, size=N)
injured = np.where((killed == 0) & (injured == 0), 1, injured)

df = pd.DataFrame({
    "Date": date.date, "Hour": hour, "State": state, "Road_Type": road, "Vehicle_Type": vehicle,
    "Cause": cause, "Weather": weather, "Killed": killed, "Injured": injured,
}).sort_values("Date").reset_index(drop=True)
df.to_csv("data/accidents.csv", index=False)
print(f"Saved {len(df):,} accident records to data/accidents.csv")
