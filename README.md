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

## Dataset Access

The dataset used is the **ERA5 Reanalysis Dataset**, provided by the European Centre for Medium-Range Weather Forecasts (ECMWF). It can be accessed via the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu).  

- **Variables used:**  
  - 10m U-component of Wind (`u10`)  
  - 2m Temperature (`t2m`)  
  - 2m Dewpoint Temperature (`d2m`)  
  - Surface Pressure (`sp`)  

- **Resolution:** 721 lat × 1440 lon grid  
- **Temporal Resolution:** 3-hour intervals  
- **Time Steps:** 64 (December 2024)  

The link to the dataset is: [Google Drive Link](https://drive.google.com/file/d/1ZNUpt32UsE8F-GpZJ2hoULh4OFhW5FL1/view?usp=sharing)


## Requirements

* Python 3.8+
* PyTorch
* GPyTorch
* NumPy
* Matplotlib

---

## Implementation Guidelines

1. **Preprocessing**  
   - Convert temperature from Kelvin → Celsius  
   - Convert pressure from Pascals → hPa  
   - Standardize features using `StandardScaler`  
   - Train/Validation/Test split: 44 / 9 / 11 time steps  

2. **Running the Framework**  
   - Step 1: Apply **GCNN** for spatial feature extraction.  
   - Step 2: Use **Mamba** for temporal dependency modeling.  
   - Step 3: Train **Gaussian Processes** on residuals for uncertainty quantification.  
   - Final prediction is a weighted ensemble (60% GCNN-Mamba, 40% GP).  

---

## Notes

- Only **pseudocode of the GCNN, Mamba, and GP modules** is provided here to highlight the core methodological contributions.  
- Full replication can be achieved by combining this pseudocode with:  
  1. The ERA5 dataset (link above), and  
  2. The preprocessing and hyperparameter details given in the manuscript.  
- This ensures reproducibility while preserving proprietary training scripts.  

---
