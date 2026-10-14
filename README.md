# Fractional-Order Crime Dynamics: Linear vs. Nonlinear Transmission Models Using Physics-Informed Neural Networks (PINNs)

[![Journal: Chaos, Solitons & Fractals](https://img.shields.io/badge/Journal-Chaos%2C%20Solitons%20%26%20Fractals%20(Elsevier)-orange.svg)](https://doi.org/10.1016/j.chaos.2026.118408)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.chaos.2026.118408-blue.svg)](https://doi.org/10.1016/j.chaos.2026.118408)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-red.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation and simulation suite for the research paper:  
**"Fractional-order crime dynamics: Linear vs. nonlinear transmission models using physics-informed neural networks"**  
*Chaos, Solitons & Fractals*, Volume 209, 2026, Article 118408.  
[https://doi.org/10.1016/j.chaos.2026.118408](https://doi.org/10.1016/j.chaos.2026.118408)

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Mathematical Formulation](#mathematical-formulation)
  - [Caputo Fractional Framework](#caputo-fractional-framework)
  - [The Four Transmission Models](#the-four-transmission-models)
- [Dynamical Systems & Stability Analysis](#dynamical-systems--stability-analysis)
  - [Basic Reproduction Number ($\mathcal{R}_0$)](#basic-reproduction-number-mathcalr_0)
  - [Equilibria & Stability](#equilibria--stability)
  - [Bifurcation & Sensitivity Analysis](#bifurcation--sensitivity-analysis)
- [Physics-Informed Neural Networks (PINNs)](#physics-informed-neural-networks-pinns)
  - [Network Architecture](#network-architecture)
  - [Caputo Derivative via Grünwald–Letnikov Discretization](#caputo-derivative-via-grünwaldletnikov-discretization)
  - [Multi-Objective Loss Formulation](#multi-objective-loss-formulation)
- [Numerical Simulation (MATLAB)](#numerical-simulation-matlab)
  - [Fractional Runge–Kutta (RK-2) Scheme](#fractional-runge-kutta-rk-2-scheme)
  - [Memory Effects ($\eta = 0.83$ vs. $\eta = 1.0$)](#memory-effects-eta--083-vs-eta--10)
- [Empirical Dataset & Results](#empirical-dataset--results)
  - [Dataset](#dataset)
  - [Estimated Parameters](#estimated-parameters)
  - [Model Fit & Out-of-Sample Forecasting](#model-fit--out-of-sample-forecasting)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Running PINN Parameter Estimation & Forecasting (Python)](#running-pinn-parameter-estimation--forecasting-python)
  - [Running Dynamical Simulations (MATLAB)](#running-dynamical-simulations-matlab)
- [Citation](#citation)
- [Authors & Acknowledgments](#authors--acknowledgments)
- [License](#license)

---

## Overview

Crime dynamics are governed by complex, nonlinear socio-economic interactions among law-abiding individuals, active offenders, and rehabilitated persons. Traditional epidemiological crime models rely on **integer-order differential equations**, implicitly assuming Markovian (memoryless) interactions with instantaneous behavioral adjustments. However, real-world criminal behavior involves **hereditary effects**, long-term social trauma, peer reinforcement, and delayed institutional responses.

This repository implements a **fractional-order mathematical modeling framework** under the **Caputo fractional derivative** ($\eta \in (0, 1]$), paired with **Physics-Informed Neural Networks (PINNs)** to:
1. Model memory-dependent transitions across susceptible non-criminals ($S$), active criminals ($C$), and prison/rehabilitated individuals ($P$).
2. Perform a rigorous comparative study between **linear (bilinear)** and **three nonlinear saturated transmission mechanisms**.
3. Learn hidden epidemiological parameters and fractional orders directly from 39 years of real-world crime data (NCRB, India).
4. Conduct out-of-sample forecasting constrained by the governing physical laws of criminal transmission.

```
                  +----------------------------------------------+
                  |           Recruitment Rate (A^η)             |
                  +----------------------------------------------+
                                         │
                                         ▼
   ┌───────────────────┐    Transmission f(S, C)    ┌───────────────────┐
   │   Non-Criminals   │ ─────────────────────────> │ Active Criminals  │
   │       S(t)        │                            │       C(t)        │
   └───────────────────┘                            └───────────────────┘
             ▲                                                │
             │ Rehabilitation                                 │ Incarceration / Law
             │ Rate (δ1^η)                                    │ Enforcement (l^η)
             │                                                ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                      Prisoners / Rehabilitated                     │
   │                                P(t)                                │
   └────────────────────────────────────────────────────────────────────┘
             │                                                │
             ▼                                                ▼
     Natural Exit (δ0^η)                              Natural Exit (δ0^η)
```

---

## Key Contributions

- **Fractional Memory Integration**: Formulates crime spread using the Caputo fractional operator of order $\eta \approx 0.83$, demonstrating that past criminal interactions significantly moderate current and future crime progression.
- **Systematic Transmission Comparison**: Evaluates bilinear contact against non-criminal saturation (preventive precautions), criminal saturation (crowding/police focus), and dual saturation.
- **Physics-Informed Parameter Identification**: Solves the inverse problem by simultaneously minimizing empirical data misfit, differential equation residuals, and initial condition losses using PyTorch autograd and Grünwald–Letnikov expansions.
- **Rigorous Mathematical Foundations**: Provides closed-form derivations of Crime-Free and Endemic equilibria, threshold reproduction numbers ($\mathcal{R}_0$), Matignon-based fractional stability criteria, center manifold transcritical bifurcations, and normalized parameter sensitivity indices.
- **Dual Verification**: Validates deep learning predictions against independent numerical simulations using a Fractional Runge–Kutta 2nd Order (RK-2) algorithm in MATLAB.

---

## Mathematical Formulation

### Caputo Fractional Framework

Let $S(t)$, $C(t)$, and $P(t)$ represent the normalized proportions of the population who are non-criminals, active criminals, and incarcerated/rehabilitated individuals, respectively. The system dynamics are governed by the Caputo fractional derivative of order $\eta \in (0, 1]$:

$${}^C \mathcal{D}_t^\eta x(t) = \frac{1}{\Gamma(1 - \eta)} \int_0^t (t - \tau)^{-\eta} x'(\tau) \, d\tau$$

To maintain dimensional consistency across fractional time, time-scaling factor $\lambda^{\eta-1}$ is incorporated:

$$\begin{aligned}
\lambda^{\eta-1} \, {}^C \mathcal{D}_t^\eta S(t) &= A^\eta - f(S, C) + \delta_1^\eta P(t) - \delta_0^\eta S(t) \\
\lambda^{\eta-1} \, {}^C \mathcal{D}_t^\eta C(t) &= f(S, C) - l^\eta C(t) - \delta_0^\eta C(t) \\
\lambda^{\eta-1} \, {}^C \mathcal{D}_t^\eta P(t) &= l^\eta C(t) - (\delta_0^\eta + \delta_1^\eta) P(t)
\end{aligned}$$

| Parameter | Sociological & Epidemiological Interpretation | Code Variable |
| :--- | :--- | :--- |
| $A$ | Baseline inflow / recruitment rate into non-criminals | `A` |
| $\alpha$ | Transmission rate of criminal behavior through contact | `alpha` |
| $\beta$ | Preventive saturation parameter (societal awareness / deterrence) | `beta` |
| $\gamma$ | Criminal saturation parameter (crowding / enforcement attention) | `gamma` |
| $l$ | Law enforcement intervention / incarceration rate | `l` |
| $\delta_0$ | Natural mortality / baseline demographic exit rate | `delta0` |
| $\delta_1$ | Rehabilitation rate returning prisoners to non-criminals | `delta1` |
| $\eta$ | Fractional derivative order ($0 < \eta \le 1$), governing memory persistence | `eta` |
| $\lambda$ | Characteristic time-scale parameter ensuring dimensional consistency | `lam` |

---

### The Four Transmission Models

The models differ in the formulation of the transmission/incidence function $f(S, C)$:

```
           Model 1: Bilinear                Model 2: Saturated in S
           f(S,C) = α^η S C                 f(S,C) = (α^η S C)/(1 + β S)
                  │                                        │
                  ├───────────────────┬────────────────────┤
                  ▼                                        ▼
       Model 3: Saturated in C              Model 4: Combined Saturation
     f(S,C) = (α^η S C)/(1 + γ C)        f(S,C) = (α^η S C)/(1 + β S + γ C)
```

#### 1. Model 1: Bilinear (Linear) Transmission
$$f(S, C) = \alpha^\eta S(t) C(t)$$
- **Assumption**: The rate of new criminal recruits is directly proportional to the product of non-criminals and criminals.
- **Limitation**: Assumes unbounded contact rates, ignoring societal protection or policing limits at large population densities.

#### 2. Model 2: Saturated Transmission in Non-Criminals ($S$)
$$f(S, C) = \frac{\alpha^\eta S(t) C(t)}{1 + \beta S(t)}$$
- **Assumption**: Incorporates a Holling Type-II saturation effect with parameter $\beta$.
- **Behavior**: As the non-criminal population grows, collective awareness, self-protective measures, and neighbourhood vigilance prevent unbounded transmission.

#### 3. Model 3: Saturated Transmission in Criminals ($C$)
$$f(S, C) = \frac{\alpha^\eta S(t) C(t)}{1 + \gamma C(t)}$$
- **Assumption**: Saturation in the criminal class with parameter $\gamma$.
- **Behavior**: At high criminal densities, police focus intensifies, criminal opportunities saturate, and peer-to-peer conversion efficiency diminishes.

#### 4. Model 4: Fully Nonlinear Combined Saturation ($S$ and $C$)
$$f(S, C) = \frac{\alpha^\eta S(t) C(t)}{1 + \beta S(t) + \gamma C(t)}$$
- **Assumption**: Beddington–DeAngelis / Crowley–Martin inspired generalized saturation.
- **Behavior**: Simultaneously models public preventive response and criminal crowding. Demonstrated to have the lowest residual error and highest empirical predictive power.

---

## Dynamical Systems & Stability Analysis

### Basic Reproduction Number ($\mathcal{R}_0$)

The basic reproduction number $\mathcal{R}_0$ dictates whether criminal behavior extinguishes or persists endemically:

- **Models 1 & 3**:
  $$\mathcal{R}_0 = \frac{\alpha^\eta S_0}{l^\eta + \delta_0^\eta} = \frac{\alpha^\eta A^\eta}{\delta_0^\eta (l^\eta + \delta_0^\eta)}$$
- **Models 2 & 4**:
  $$\mathcal{R}_0 = \frac{\alpha^\eta S_0}{(1 + \beta S_0)(l^\eta + \delta_0^\eta)} = \frac{\alpha^\eta A^\eta}{(\delta_0^\eta + \beta A^\eta)(l^\eta + \delta_0^\eta)}$$

### Equilibria & Stability

1. **Crime-Free Equilibrium (CFE)**:
   $$E_0 = \left(S_0, 0, 0\right) = \left(\frac{A^\eta}{\delta_0^\eta}, 0, 0\right)$$
   By **Matignon’s Stability Criterion** for fractional systems, $E_0$ is locally asymptotically stable if:
   $$|\arg(\lambda_i)| > \frac{\eta \pi}{2} \quad \Longleftrightarrow \quad \mathcal{R}_0 < 1$$
   If $\mathcal{R}_0 > 1$, $E_0$ becomes unstable and crime persists.

2. **Endemic Equilibrium (EE)**:
   $$E^* = (S^*, C^*, P^*), \quad C^* > 0$$
   Exists uniquely in the positive invariant region whenever $\mathcal{R}_0 > 1$.

### Bifurcation & Sensitivity Analysis

- **Transcritical Bifurcation**: The system undergoes a forward transcritical bifurcation at $\mathcal{R}_0 = 1$ (verified via Center Manifold Theory). An exchange of stability occurs: for $\mathcal{R}_0 < 1$, $E_0$ is stable; as $\mathcal{R}_0$ crosses 1, $E_0$ loses stability and $E^*$ becomes stable.
- **Normalized Sensitivity Indices ($\Upsilon_p^{\mathcal{R}_0} = \frac{\partial \mathcal{R}_0}{\partial p} \frac{p}{\mathcal{R}_0}$)**:
  - $\frac{\partial \mathcal{R}_0}{\partial A} > 0$ and $\frac{\partial \mathcal{R}_0}{\partial \alpha} > 0$ (Increasing inflow or transmission inflates crime reproduction).
  - $\frac{\partial \mathcal{R}_0}{\partial \beta} < 0$, $\frac{\partial \mathcal{R}_0}{\partial l} < 0$, $\frac{\partial \mathcal{R}_0}{\partial \delta_0} < 0$ (Social resistance, proactive law enforcement, and natural turnover suppress crime propagation).

---

## Physics-Informed Neural Networks (PINNs)

Unlike purely statistical time-series or standard black-box neural networks, PINNs incorporate the fractional differential equations directly into the training loss function.

```
               Collocation Time Points t ∈ [0, 1]
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   Multi-Layer Perceptron (tanh)     │
            │          [1 → 64 → 64 → 3]           │
            └──────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
          S_pred             C_pred             P_pred
            │                  │                  │
            ├──────────────────┼──────────────────┤
            │                  │                  │
            ▼                  ▼                  ▼
    Grünwald–Letnikov  Residual Evaluator    Data Fitting Loss
      Frac-Deriv D^η   (λ^(η-1) D^η - RHS)     MSE(P_pred, P_obs)
            │                  │                  │
            └──────────────────┬──────────────────┘
                               │
                               ▼
                   Total Composite Loss Function
        L = w_phys * L_phys + w_data * L_data + w_ic * L_ic
                               │
                               ▼
                      Adam Optimization
              (Updates Network Weights & System
                Parameters: A, α, β, γ, l, δ, η)
```

### Network Architecture

- **Input Layer**: Dimension 1 (normalized time $t \in [0, 1]$).
- **Hidden Layers**: 2 fully connected hidden layers with 64 units each and $\tanh$ activation.
- **Output Layer**: 3 units mapped through $\text{sigmoid}$ activation to enforce physical non-negativity and boundedness ($S, C, P \in [0, 1]$).
- **Learnable Parameters**: Base parameter values with bounded learnable perturbations optimized concurrently with the neural network weights.

### Caputo Derivative via Grünwald–Letnikov Discretization

The fractional derivative in the autograd computation graph is evaluated via the Grünwald–Letnikov (GL) expansion:

$${}^C \mathcal{D}_t^\eta u(t_n) \approx \frac{1}{h^\eta} \sum_{j=0}^n w_j^{(\eta)} u(t_{n-j}), \quad w_j^{(\eta)} = (-1)^j \binom{\eta}{j} = \frac{(-1)^j \Gamma(\eta + 1)}{\Gamma(j + 1)\Gamma(\eta - j + 1)}$$

Implemented using numerically stable log-gamma representations (`torch.lgamma`).

### Multi-Objective Loss Formulation

$$\mathcal{L}_{\text{total}} = w_{\text{phys}} \mathcal{L}_{\text{phys}} + w_{\text{data}} \mathcal{L}_{\text{data}} + w_{\text{ic}} \mathcal{L}_{\text{ic}}$$

Where:
- **Physics Loss ($\mathcal{L}_{\text{phys}}$)**: Residual mean-squared error enforcing governing equations across all time steps:
  $$\mathcal{L}_{\text{phys}} = \frac{1}{N-1} \sum_{n=1}^{N-1} \left[ \left(\lambda^{\eta-1} \mathcal{D}^\eta S - \text{RHS}_S\right)^2 + \left(\lambda^{\eta-1} \mathcal{D}^\eta C - \text{RHS}_C\right)^2 + \left(\lambda^{\eta-1} \mathcal{D}^\eta P - \text{RHS}_P\right)^2 \right]$$
- **Data Loss ($\mathcal{L}_{\text{data}}$)**: Mean-squared error against normalized historical crime observations:
  $$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{n=0}^{N-1} \left(P_{\text{pred}}(t_n) - P_{\text{obs}}(t_n)\right)^2$$
- **Initial Condition Loss ($\mathcal{L}_{\text{ic}}$)**:
  $$\mathcal{L}_{\text{ic}} = (S(0) - S_0)^2 + (C(0) - C_0)^2 + (P(0) - P_0)^2$$
- **Default Hyperparameter Weights**: $w_{\text{phys}} = 1.0$, $w_{\text{data}} = 40.0$, $w_{\text{ic}} = 200.0$.

---

## Numerical Simulation (MATLAB)

To independently verify the PINN results, the fractional system is integrated using a modified **Fractional Runge–Kutta 2nd Order (RK-2)** algorithm.

### Fractional Runge–Kutta (RK-2) Scheme

Given step size $h$ and time grid $t_k = k h$:

$$\begin{aligned}
k_1 &= f(t_k, X_k) \\
X_{\text{mid}} &= X_k + \frac{h^\eta}{2 \Gamma(\eta + 1)} k_1 \\
k_2 &= f\left(t_k + \frac{h}{2}, X_{\text{mid}}\right) \\
X_{k+1} &= X_k + \frac{h^\eta}{\Gamma(\eta + 1)} k_2
\end{aligned}$$

Convergence was rigorously evaluated with step sizes $h \in \{0.02, 0.01, 0.005, 0.0025\}$, confirming monotonic $L_2$ error decay.

### Memory Effects ($\eta = 0.83$ vs. $\eta = 1.0$)

- **Fractional Order ($\eta = 0.83$)**: Slower decay, persistent memory of past crime peaks, realistic social inertia.
- **Classical Integer Order ($\eta = 1.0$)**: Instantaneous exponential response without hereditary persistence, underestimating long-term endemic levels.

---

## Empirical Dataset & Results

### Dataset

The models were calibrated using official empirical data published by the **National Crime Records Bureau (NCRB)**, Ministry of Home Affairs, Government of India:
- **Total Cognizable Crimes**: 1982 to 2020 (39 annual data points).
- **Observed Range**: From 1.74 Million crimes (1982) up to 3.52 Million crimes (2019) and 2.93 Million (2020).

### Estimated Parameters

Parameters identified by the PINN framework across all four models:

| Parameter | Model 1 (Linear) | Model 2 (Sat. in S) | Model 3 (Sat. in C) | Model 4 (Combined) |
| :--- | :---: | :---: | :---: | :---: |
| $A$ (Inflow rate) | 0.060089 | 0.060089 | 0.060089 | **0.060090** |
| $\alpha$ (Transmission rate) | 0.699931 | 0.699931 | 0.699929 | **0.699931** |
| $\beta$ (Saturation in $S$) | — | 0.200078 | — | **0.200079** |
| $\gamma$ (Saturation in $C$) | — | — | 0.300013 | **0.300003** |
| $l$ (Law enforcement) | 0.090095 | 0.090095 | 0.090095 | **0.090095** |
| $\delta_0$ (Natural removal) | 0.069907 | 0.069907 | 0.069907 | **0.069906** |
| $\delta_1$ (Rehabilitation) | 0.029904 | 0.029904 | 0.029904 | **0.029904** |
| $\eta$ (Fractional order) | 0.830094 | 0.830094 | 0.830094 | **0.830094** |
| $\lambda$ (Scaling parameter) | 1.600093 | 1.600093 | 1.600093 | **1.600093** |

### Model Fit & Out-of-Sample Forecasting

#### Training Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

| Model | Specification | Training MAPE |
| :--- | :--- | :---: |
| **Model 1** | Bilinear (Linear) | 1.18% |
| **Model 2** | Saturated in Non-Criminals ($S$) | 1.30% |
| **Model 3** | Saturated in Criminals ($C$) | 1.18% |
| **Model 4** | **Combined Nonlinear Saturation ($S$ and $C$)** | **1.17% (Best)** |

#### Out-of-Sample Prediction Accuracy (Validation on 2021 & 2022 Actual Data)

| Metric / Year | Actual NCRB Data | Model 1 (Linear) | Model 2 (Sat. S) | Model 3 (Sat. C) | Model 4 (Combined) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2021 Predicted** | 3.6634 Million | 3.3867 M | 3.3879 M | 3.3867 M | **3.3866 M** |
| **2021 Error** | — | 0.2611 M | 0.2628 M | 0.2619 M | **0.2618 M** |
| **2021 Accuracy** | — | 92.87% | 92.82% | 92.84% | **92.85%** |
| **2022 Predicted** | 3.5614 Million | 3.3977 M | 3.3991 M | 3.3978 M | **3.3976 M** |
| **2022 Error** | — | 0.1489 M | 0.1502 M | 0.1495 M | **0.1486 M** |
| **2022 Accuracy** | — | 95.82% | 95.78% | 95.80% | **95.83% (Best)** |
| **Net Accuracy** | — | **94.34%** | 94.30% | 94.32% | **94.34%** |

> **Conclusion**: Model 4 consistently delivers the lowest residual error, smallest training MAPE, and highest out-of-sample prediction accuracy.

---

## Repository Structure

```
Fractional-Order-Crime-Dynamics-Linear-vs-Nonlinear-Transmission-Models-Using-PINNS/
├── LICENSE                                        # MIT License
├── README.md                                      # Documentation & usage guide
├── requirements.txt                               # Python dependencies
│
├── MATLAB/                                        # Numerical simulation suite
│   ├── Models/
│   │   ├── Model_1.m                              # Fractional RK-2 for Model 1 (Linear)
│   │   ├── Model_2.m                              # Fractional RK-2 for Model 2 (Saturated in S)
│   │   ├── Model_3.m                              # Fractional RK-2 for Model 3 (Saturated in C)
│   │   └── Model_4.m                              # Fractional RK-2 for Model 4 (Dual saturation)
│   └── Graphs/                                    # Exported vector EPS plots
│       ├── Crime1_1.eps, Crime1_2.eps
│       ├── Crime2_1.eps, Crime2_2.eps
│       ├── Crime3_1.eps, Crime3_2.eps
│       └── Crime4_1.eps, Crime4_2.eps
│
└── Python_Codes_For_Forecasting_And_Parameter_Estimation/
    ├── Models/
    │   ├── Model_1.py                             # PINN for Model 1 (Linear transmission)
    │   ├── Model_2.py                             # PINN for Model 2 (Saturation in non-criminals)
    │   ├── Model_3.py                             # PINN for Model 3 (Saturation in criminals)
    │   └── Model_4.py                             # PINN for Model 4 (Combined saturation)
    └── Graphs/                                    # Generated PNG loss curves & forecast plots
        ├── Crime1.png, Crime1Loss.png
        ├── Crime2.png, Crime2Loss.png
        ├── Crime3.png, Crime3Loss.png, Crime3_1.png
        └── Crime4.png, Crime4Loss.png
```

---

## Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshitPranjal/Fractional-Order-Crime-Dynamics-Linear-vs-Nonlinear-Transmission-Models-Using-PINNS.git
   cd Fractional-Order-Crime-Dynamics-Linear-vs-Nonlinear-Transmission-Models-Using-PINNS
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running PINN Parameter Estimation & Forecasting (Python)

To run the PINN training, parameter identification, and 2021–2024 forecasting for the best-performing model (Model 4):

```bash
python Python_Codes_For_Forecasting_And_Parameter_Estimation/Models/Model_4.py
```

To run any of the other models:
```bash
python Python_Codes_For_Forecasting_And_Parameter_Estimation/Models/Model_1.py  # Linear
python Python_Codes_For_Forecasting_And_Parameter_Estimation/Models/Model_2.py  # Saturation in S
python Python_Codes_For_Forecasting_And_Parameter_Estimation/Models/Model_3.py  # Saturation in C
```

**Expected Output**:
- Real-time training progress displayed every 1,000 epochs.
- Table of estimated system parameters ($A, \alpha, \beta, \gamma, l, \delta_0, \delta_1, \eta, \lambda$).
- Training MAPE percentage.
- Projected crime figures for 2021, 2022, 2023, and 2024.
- High-resolution loss convergence curve and fit/forecast plots saved locally.

### Running Dynamical Simulations (MATLAB)

1. Open MATLAB (R2021a or later).
2. Set the working directory to `MATLAB/Models/`.
3. Execute any script from the command prompt or editor:
   ```matlab
   run('Model_4.m')
   ```
4. The script computes the fractional trajectory using Fractional RK-2 and displays the time evolution of Susceptible ($S$), Criminally Active ($C$), and Prison ($P$) populations over $t \in [0, 20]$.

---

## Citation

If you find this work, codebase, or methodology useful in your research, please cite our paper:

```bibtex
@article{Bansal2026ChaosSolitons,
  title   = {Fractional-order crime dynamics: Linear vs. nonlinear transmission models using physics-informed neural networks},
  author  = {Bansal, Komal and Pranjal, Harshit and Yadav, Poonam and Mathur, Trilok and Agarwal, Shivi},
  journal = {Chaos, Solitons \& Fractals},
  volume  = {209},
  pages   = {118408},
  year    = {2026},
  issn    = {0960-0779},
  doi     = {10.1016/j.chaos.2026.118408},
  url     = {https://doi.org/10.1016/j.chaos.2026.118408}
}
```

---

## Authors & Acknowledgments

- **Komal Bansal** — *Department of Mathematics, School of Advanced Sciences, VIT-AP University, Amaravati, 522237, India*
- **Harshit Pranjal** — *Department of Mathematics, Birla Institute of Technology, Mesra, Ranchi, India*
- **Poonam Yadav** — *Department of Mathematics, Birla Institute of Technology and Science, Pilani, 333031, India*
- **Trilok Mathur** — *Department of Mathematics, Birla Institute of Technology and Science, Pilani, 333031, India*
- **Shivi Agarwal** — *Department of Mathematics, Birla Institute of Technology and Science, Pilani, 333031, India*

We gratefully acknowledge the **National Crime Records Bureau (NCRB)**, Ministry of Home Affairs, Government of India, for maintaining the public crime records that facilitated this empirical investigation.

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

