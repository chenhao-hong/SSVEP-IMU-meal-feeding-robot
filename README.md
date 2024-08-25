# SSVEP-Based BCI System for Robotic Arm Control

## Project Overview
This repository contains the code and documentation for a Brain-Computer Interface (BCI) system designed to assist individuals with motor impairments. The system utilizes Steady-State Visual Evoked Potentials (SSVEP) to enable users to control a robotic arm for self-feeding by simply fixating on specific visual stimuli. This project integrates Canonical Correlation Analysis (CCA) for feature extraction and employs Inertial Measurement Unit (IMU) sensors to enhance control accuracy and reduce false positives.

## System Description
The BCI system presented in this project allows users to command a robotic arm to scoop and deliver food through thought alone, enhancing the quality of life for individuals with disabilities. Our experimental results confirm the system's effectiveness and accuracy, demonstrating its potential in real-world applications.

## Equipment Used
The experiments were conducted using the Unicorn Hybrid Black EEG cap, which is instrumental in capturing EEG data for processing. 

## Getting Started
To set up and run this project, please follow the guidelines below:

### Prerequisites
Ensure you have the following installed:
- Python 3.11
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `pygame`, `scikit-learn`, `scipy`
- Unicorn's official Python API (See [Unicorn's documentation](https://unicorn-bi.com/))

### Installation
1. Clone this repository:
https://github.com/chenhao-hong/SSVEP-IMU-meal-feeding-robot.git
2. Install the required Python packages:
pip install -r requirements.txt
### Running the System
To start the system, navigate to the project directory and run:
python main.py

## Documentation
For more details on the implementation and usage of the system, refer to the [official tutorial](https://unicorn-bi.com/) provided by the manufacturers of the Unicorn Hybrid Black EEG cap.

