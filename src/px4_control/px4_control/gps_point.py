#!/usr/bin/env python3

import json
import os
import sys

"""
Script Name: gps_point.py
Description: The GPS point saved by record_drone_state.py and flown to by the obstacle_*.py scripts (no ROS code).
             It is one file per computer, so it does not matter from which folder or workspace they are started.

Last Modified: 2026-11-09
"""

GPS_POINT_FILE = os.path.expanduser("~/.config/px4_control/gps_point.json")


def load_gps_point(filename=None):
    """(lat, lon, alt) of the first state in a .json file made by record_drone_state.py, by default GPS_POINT_FILE"""
    if filename is None:
        filename = GPS_POINT_FILE
    elif not os.path.splitext(filename)[1]:
        filename += ".json"
    try:
        with open(filename) as f:
            states = json.load(f)
    except FileNotFoundError:
        sys.exit(f"{filename} does not exist. Save a GPS point first: ros2 run px4_control record_drone_state")
    if len(states) != 1:
        print(f"Warning: {len(states)} states in {filename}, using the first one")
    print(f"GPS point from {filename}" + (f", saved {states[0]['time']}" if "time" in states[0] else ""))
    gp = states[0]["global_position"]
    return gp["lat"], gp["lon"], gp["alt"]
