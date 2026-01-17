"""
CPU Thermal ODE Simulator
Programmer(s): Jared Walker
Packages: numpy, matplotlib, scipy
Purpose/Approach:
Solve and visualize a CPU thermal model (an ODE) under a workload/power schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@dataclass
class ThermalParams:
    C: float          # Thermal capacitance (J/°C)
    k: float          # Cooling coefficient (W/°C)
    T_amb: float      # Ambient temperature (°C)


@dataclass
class PowerSegment:
    t_start: float    # seconds
    t_end: float      # seconds
    P: float          # watts


def read_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return float(default)
    return float(raw)


def read_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return int(default)
    return int(raw)


def build_default_schedule(tf: float) -> List[PowerSegment]:
    # Idle -> load -> idle
    t1 = min(10.0, tf * 0.2)
    t2 = min(40.0, tf * 0.7)
    return [
        PowerSegment(0.0, t1, 15.0),
        PowerSegment(t1, t2, 75.0),
        PowerSegment(t2, tf, 20.0),
    ]


def read_power_schedule(tf: float) -> List[PowerSegment]:
    mode = input("Use default power schedule? (y/n) [y]: ").strip().lower()
    if mode in ("", "y", "yes"):
        return build_default_schedule(tf)

    n = read_int("How many power segments?", 3)
    segments: List[PowerSegment] = []

    print("Enter segments as: start_time end_time power_watts (times in seconds)")
    for i in range(n):
        while True:
            raw = input(f"Segment {i+1}: ").strip()
            parts = raw.split()
            if len(parts) != 3:
                print("Please enter exactly 3 values: start end power")
                continue
            t_start, t_end, p = map(float, parts)
            if t_start < 0 or t_end <= t_start or t_end > tf:
                print(f"Invalid times. Require 0 <= start < end <= tf={tf}.")
                continue
            segments.append(PowerSegment(t_start, t_end, p))
            break

    segments.sort(key=lambda s: s.t_start)
    if segments[0].t_start != 0.0:
        raise ValueError("Schedule must start at time 0.0.")
    if abs(segments[-1].t_end - tf) > 1e-9:
        raise ValueError(f"Last segment must end exactly at tf={tf}.")
    for a, b in zip(segments, segments[1:]):
        if abs(a.t_end - b.t_start) > 1e-9:
            raise ValueError("Segments must be contiguous (end time must match next start time).")

    return segments


def power_at_time(t: float, schedule: List[PowerSegment]) -> float:
    # Assumes contiguous schedule covering [0, tf]
    for seg in schedule:
        if seg.t_start <= t < seg.t_end:
            return seg.P
    return schedule[-1].P


def cpu_thermal_rhs(t: float, T: np.ndarray, params: ThermalParams, schedule: List[PowerSegment]) -> np.ndarray:
    # dT/dt = (P(t) - k*(T - Tamb)) / C
    P = power_at_time(t, schedule)
    dTdt = (P - params.k * (T[0] - params.T_amb)) / params.C
    return np.array([dTdt], dtype=float)


def solve_temperature(
        T0: float,
        tf: float,
        params: ThermalParams,
        schedule: List[PowerSegment],
        rtol: float,
        atol: float,
        n_eval: int,
) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(0.0, tf, n_eval)
    sol = solve_ivp(
        fun=lambda t, y: cpu_thermal_rhs(t, y, params, schedule),
        t_span=(0.0, tf),
        y0=np.array([T0], dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.t, sol.y[0]


def estimate_error_coarse_vs_fine(
        T0: float,
        tf: float,
        params: ThermalParams,
        schedule: List[PowerSegment],
) -> float:
    # Simple error estimate: solve with looser vs tighter tolerances and compare on same time grid
    t1, y1 = solve_temperature(T0, tf, params, schedule, rtol=1e-5, atol=1e-7, n_eval=600)
    t2, y2 = solve_temperature(T0, tf, params, schedule, rtol=1e-8, atol=1e-10, n_eval=600)

    # max absolute difference as a rough estimate (°C)
    return float(np.max(np.abs(y2 - y1)))


def plot_results(t: np.ndarray, T: np.ndarray, schedule: List[PowerSegment], params: ThermalParams) -> None:
    plt.figure()
    plt.plot(t, T)

    for seg in schedule[:-1]:
        plt.axvline(seg.t_end, linestyle="--")

    plt.title("CPU Temperature vs Time (Thermal ODE Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.show()


def summarize(t: np.ndarray, T: np.ndarray, schedule: List[PowerSegment]) -> None:
    idx_max = int(np.argmax(T))
    print("\n--- Summary ---")
    print(f"Final temperature: {T[-1]:.3f} °C at t={t[-1]:.2f} s")
    print(f"Peak temperature : {T[idx_max]:.3f} °C at t={t[idx_max]:.2f} s")
    print("Power schedule (W):")
    for seg in schedule:
        print(f"  {seg.t_start:.2f}–{seg.t_end:.2f} s : {seg.P:.2f} W")


def main() -> None:
    print("CPU Thermal ODE Simulator")
    print("Model: dT/dt = (P(t) - k*(T - Tamb)) / C\n")

    tf = read_float("Sim end time tf (seconds)", 60.0)
    T0 = read_float("Initial CPU temperature T(0) (°C)", 35.0)
    T_amb = read_float("Ambient temperature Tamb (°C)", 22.0)
    C = read_float("Thermal capacitance C (J/°C)", 120.0)
    k = read_float("Cooling coefficient k (W/°C)", 2.5)

    params = ThermalParams(C=C, k=k, T_amb=T_amb)
    schedule = read_power_schedule(tf)

    t, T = solve_temperature(
        T0=T0,
        tf=tf,
        params=params,
        schedule=schedule,
        rtol=1e-6,
        atol=1e-8,
        n_eval=600,
    )

    err = estimate_error_coarse_vs_fine(T0, tf, params, schedule)
    summarize(t, T, schedule)
    print(f"Estimated numerical error (max |ΔT|): ~{err:.6f} °C")

    plot_results(t, T, schedule, params)


if __name__ == "__main__":
    main()
