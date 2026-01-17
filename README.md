# CPU Thermal ODE Simulator

## Description
This program simulates CPU temperature behavior using an ordinary differential equation.
It models heat generation from CPU power usage and heat dissipation to the environment.

## Requirements
- Python 3.9+
- numpy
- matplotlib
- scipy

## Installation
Install dependencies using pip:

pip install numpy matplotlib scipy

## How to Run
Run the program from the terminal:

python cpu_thermal_ode.py

Follow the prompts to enter:
- Simulation time
- Initial temperature
- Ambient temperature
- Thermal parameters
- Power schedule (default or custom)

## Output
- Terminal summary of results
- Estimated numerical error
- Plot of CPU temperature vs time
