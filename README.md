# GCNN-Mamba-GP

This repository contains the core modules used in the research project:
**Probabilistic Weather Forecasting using a Hybrid GCNN-Mamba-GP Framework for Enhanced Spatiotemporal Accuracy**

The model integrates:

* **GCNN (Gated Convolutional Neural Network)** for spatial pattern extraction,
* **Mamba (Structured State-Space Model)** for efficient temporal sequence modeling,
* **Multitask Gaussian Process (GP)** for residual correction and uncertainty quantification.

---

## Repository Structure

| File / Folder              | Description                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `GCNN_Mamba_GP.py`         | Pseudocode of implementation.                                                        |
| `README.md`                | This documentation file.                                                             |

---

## Core Idea

A **hybrid weather forecasting model** is proposed which:

1. Uses **GCNN** to extract fine-grained spatial patterns over meteorological grids.
2. Applies **Mamba** to learn long-range temporal dependencies efficiently.
3. Trains a **Gaussian Process** to model residuals or corrections and quantify predictive uncertainty.

This modular approach enables high accuracy with interpretable, confidence-aware forecasts — crucial for real-world atmospheric systems.

---

## Dataset Overview

* **Source:** ERA5 reanalysis dataset (December 2024)
* **Link:** [Link to dataset](https://drive.google.com/file/d/1ZNUpt32UsE8F-GpZJ2hoULh4OFhW5FL1/view?usp=sharing)
* **Variables:**

  * 10m U-component of Wind (`u10`)
  * 2m Temperature (`t2m`)
  * 2m Dewpoint Temperature (`d2m`)
  * Surface Pressure (`sp`)
* **Spatial Resolution:** 721 lat × 1440 lon grid
* **Temporal Resolution:** 3-hour intervals (sampled every 3rd day)
* **Total Time Steps:** 64 (spanning December)

---

## Requirements

* Python 3.8+
* PyTorch
* GPyTorch
* NumPy
* Matplotlib

