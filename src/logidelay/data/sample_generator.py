from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd


CITY_COORDINATES = {
    "Memphis": (35.1495, -90.0490),
    "Chicago": (41.8781, -87.6298),
    "Dallas": (32.7767, -96.7970),
    "Atlanta": (33.7490, -84.3880),
    "Newark": (40.7357, -74.1724),
}


def _jitter_coordinate(value: float, spread: float = 0.08) -> float:
    """Create nearby synthetic points around a city center."""
    return value + random.uniform(-spread, spread)


def generate_sample_logistics_events(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic country-neutral logistics event logs for MVP development.

    This is not the final research dataset. It allows us to build and test the
    diagnosis, severity, distance-aware route instability, and explanation pipeline
    before integrating the real public benchmark dataset.
    """
    random.seed(seed)

    cities = list(CITY_COORDINATES.keys())
    zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    task_types = ["delivery", "pickup"]

    rows = []
    base_time = datetime(2026, 1, 1, 8, 0, 0)

    for i in range(n_rows):
        package_id = f"PKG-{100000 + i}"
        courier_id = f"CR-{random.randint(100, 140)}"
        city = random.choice(cities)
        zone_id = random.choice(zones)
        task_type = random.choice(task_types)

        city_lat, city_lng = CITY_COORDINATES[city]

        origin_lat = _jitter_coordinate(city_lat)
        origin_lng = _jitter_coordinate(city_lng)
        destination_lat = _jitter_coordinate(city_lat)
        destination_lng = _jitter_coordinate(city_lng)

        assigned_time = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 10),
            minutes=random.randint(0, 59),
        )

        scenario = random.choices(
            population=[
                "normal",
                "acceptance_delay",
                "pickup_delay",
                "execution_delay",
                "workload_pressure",
                "event_inconsistency",
                "severe_time_window_violation",
            ],
            weights=[45, 12, 10, 12, 10, 5, 6],
            k=1,
        )[0]

        if scenario == "acceptance_delay":
            accept_gap = random.randint(75, 180)
            pickup_gap = random.randint(10, 45)
            execution_minutes = random.randint(80, 180)
            promised_buffer = random.randint(120, 220)
            workload = random.randint(5, 12)

        elif scenario == "pickup_delay":
            accept_gap = random.randint(10, 35)
            pickup_gap = random.randint(90, 220)
            execution_minutes = random.randint(80, 160)
            promised_buffer = random.randint(150, 260)
            workload = random.randint(4, 10)

        elif scenario == "execution_delay":
            accept_gap = random.randint(10, 35)
            pickup_gap = random.randint(10, 50)
            execution_minutes = random.randint(240, 480)
            promised_buffer = random.randint(160, 260)
            workload = random.randint(4, 10)

        elif scenario == "workload_pressure":
            accept_gap = random.randint(35, 90)
            pickup_gap = random.randint(30, 90)
            execution_minutes = random.randint(160, 300)
            promised_buffer = random.randint(150, 260)
            workload = random.randint(13, 25)

        elif scenario == "event_inconsistency":
            accept_gap = random.randint(10, 40)
            pickup_gap = random.randint(10, 40)
            execution_minutes = random.randint(80, 180)
            promised_buffer = random.randint(150, 260)
            workload = random.randint(3, 10)

        elif scenario == "severe_time_window_violation":
            accept_gap = random.randint(50, 150)
            pickup_gap = random.randint(40, 150)
            execution_minutes = random.randint(360, 720)
            promised_buffer = random.randint(120, 240)
            workload = random.randint(8, 18)

        else:
            accept_gap = random.randint(5, 30)
            pickup_gap = random.randint(5, 30)
            execution_minutes = random.randint(45, 140)
            promised_buffer = random.randint(180, 360)
            workload = random.randint(2, 8)

        accepted_time = assigned_time + timedelta(minutes=accept_gap)
        pickup_time = accepted_time + timedelta(minutes=pickup_gap)
        completed_time = pickup_time + timedelta(minutes=execution_minutes)
        promised_delivery_time = assigned_time + timedelta(minutes=promised_buffer)

        if scenario == "event_inconsistency":
            completed_time_value = None
        else:
            completed_time_value = completed_time

        rows.append(
            {
                "package_id": package_id,
                "courier_id": courier_id,
                "city": city,
                "zone_id": zone_id,
                "task_type": task_type,
                "origin_lat": origin_lat,
                "origin_lng": origin_lng,
                "destination_lat": destination_lat,
                "destination_lng": destination_lng,
                "assigned_time": assigned_time,
                "accepted_time": accepted_time,
                "pickup_time": pickup_time,
                "completed_time": completed_time_value,
                "promised_delivery_time": promised_delivery_time,
                "courier_workload_2h": workload,
                "scenario_source": scenario,
            }
        )

    return pd.DataFrame(rows)