#!/usr/bin/env python3
"""Compatibility entry point for the pickup inference implementation."""
import runpy
import sys
from pathlib import Path
sys.argv.insert(1, "infer")
runpy.run_path(str(Path(__file__).with_name("pickup_robot.py")),run_name="__main__")
