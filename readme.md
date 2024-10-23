# FractalGrid Microgrid Simulation using DQN

## Overview
This project implements a simulation of a fractal grid consisting of multiple microgrids interconnected by relays, designed to optimize power management using **Deep Q-Learning (DQN)**. Each microgrid has solar panels, energy storage systems (ESS), and connections to the main grid. The objective is to minimize the total operational cost, including energy transfers between microgrids and with the main grid.

## Features
- **DQN-based optimization** for microgrid energy management
- Real-time PV dispatch, battery SOC tracking, and grid interactions
- Visualization of switching states, net load, power transfers, and more
- Custom Gym environment for RL training

## Setup

### Dependencies
Install the required packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### License
This project is licensed under the MIT License.