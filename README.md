# A Visual Tour of Optimization Pathologies

This repository contains a first-principles numerical and visual analysis of why simple optimization is hard. Using only Python and NumPy, we build a "compass" to probe the geometry of 2D loss landscapes and use it to conduct a definitive experiment that reveals the catastrophic failure modes of vanilla Stochastic Gradient Descent (SGD) and the stabilizing power of Momentum.

The entire project is self-contained in a single, heavily narrated Jupyter Notebook: [`A_Visual_Tour.ipynb`](./Analysis.ipynb).

---

## The Core Finding: Stability is More Than Speed

The central experiment of this project is a "knockout" test designed to reveal the fundamental difference between SGD and Momentum. We find a "Goldilocks" learning rate that is high enough to cause SGD to catastrophically fail, but which Momentum can handle with ease.

This proves a deep insight: **Momentum's primary benefit is not just speed, but stability.** It fundamentally widens the range of viable hyperparameters by making the optimization process more robust to the pathological curvature of the loss landscape.

### The "Knockout" Plot: SGD (Fails) vs. Momentum (Succeeds)

<img width="1023" height="778" alt="image" src="https://github.com/user-attachments/assets/2e67d7a0-c960-4526-8952-3dfcd1222732" />
 

As the plot clearly shows, at the same aggressive learning rate:
*   **SGD (Red Path)** violently oscillates across the steep walls of the Rosenbrock valley and fails to converge.
*   **Momentum (Green Path)** successfully dampens these oscillations, accumulating velocity along the valley floor and smoothly converging to the global minimum.

---

## Methodology from First Principles

This project is an exercise in deconstruction. We do not rely on automatic differentiation libraries.

1.  **Numerical Engine:** We build a `LossSurface` class with methods to compute the **Gradient** and the **Hessian** matrix at any point using the central finite difference method. This provides a direct, numerical look at the local slope and curvature.

2.  **Geometric Analysis:** We use the eigendecomposition of the Hessian to numerically analyze the landscape's geometry. This allows us to precisely quantify the **pathological curvature** (anisotropy) of functions like the Rosenbrock, proving *why* they are difficult to optimize.
    *   **At the Valley Wall:** `Condition Number ≈ 200+` (Extremely anisotropic)
    *   **At the Valley Floor:** `Condition Number ≈ 100+` (Still highly anisotropic)

3.  **Optimizer Simulation:** We implement simple `sgd` and `momentum` optimizers from scratch to simulate their trajectories on the loss surface, directly producing the paths seen in the plots.

## Usage

1.  Clone this repository:
    ```bash
    git clone https://github.com/S-Sairam/loss-landscape-calculus.git
    cd loss-landscape-calculus
    ```
2.  Ensure you have the required libraries:
    ```bash
    pip install numpy matplotlib jupyter
    ```
3.  Launch Jupyter Notebook and open `Analysis.ipynb`.
4.  Run all cells to reproduce the analysis and generate the plots.

---
