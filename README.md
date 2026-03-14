# SENTRY AI

Satellite Encounter Analysis & Autonomous Risk-Mitigation System

## Overview

SENTRY AI is a research-oriented system for **space situational awareness and satellite collision risk prediction**.
It simulates satellite orbits, predicts conjunction events, and estimates collision probability using orbital mechanics and statistical models.

## Features

* Real-time **satellite orbit simulation**
* Uses **SGP4 propagation model**
* Processes **TLE satellite catalog**
* Covariance-based orbit uncertainty modeling
* **Mahalanobis distance collision probability estimation**
* 3D orbit visualization
* Synthetic conjunction scenario generator
* Automated collision avoidance burn logic

## Tech Stack

* Python
* NumPy
* SciPy
* Orbital Mechanics Libraries
* Matplotlib / 3D Visualization

## System Workflow

1. Load TLE satellite data
2. Propagate satellite orbits using SGP4
3. Model orbital uncertainty using covariance matrices
4. Detect potential conjunction events
5. Estimate collision probability
6. Trigger autonomous maneuver decision logic

## How to Run

Clone the repository

```bash
git clone https://github.com/Sasi934/SENTRY-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the simulation

```bash
python main.py
```

## Future Enhancements

* Real-time satellite tracking APIs
* Reinforcement learning for maneuver planning
* Integration with real space situational awareness datasets
* Advanced 3D orbit visualization engine

## Author

**Veera Shashank Reddy Ippala**
