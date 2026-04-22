# Optimizer Visualizer — CS 566

An interactive Streamlit app for exploring how different optimization algorithms behave on common loss surfaces used in machine learning and numerical optimization.

## Features

- Compare **SGD**, **Momentum**, **Nesterov**, **RMSProp**, and **AdamW**
- Visualize optimizer trajectories on multiple benchmark loss surfaces
- Explore **Hessian-based local curvature** with a 2D contour view and local 3D patch
- Study **stochastic noise effects** and mini-batch behavior
- Analyze **saddle-point escape**
- Inspect **convergence curves**
- Track **optimizer internals** such as:
  - gradient norm
  - step norm
  - effective learning rate
  - momentum / velocity state
  - first- and second-moment statistics
- Use **preset experiments** for reproducible demonstrations:
  - Ill-conditioned valley
  - Noisy training
  - Saddle escape
  - Multi-modal landscape

## Demo Purpose

This app was built for **CS 566: Math for AI** as an educational tool to help explain why optimizers behave differently under changing curvature, noise, and geometry.

## App Structure

The app includes the following main sections:

- **3D Surface** — optimizer trajectories over the loss surface
- **Hessian** — local gradient, eigenvectors, and curvature analysis
- **Stochastic** — noise-driven trajectory comparisons
- **Saddle Escape** — optimizer behavior near saddle regions
- **Convergence** — loss vs. iteration plots
- **Optimizer Internals** — hidden state analysis across optimizers

## Supported Loss Surfaces

- Rosenbrock
- Beale
- Saddle Valley
- Himmelblau
- Rastrigin
- Ackley
- Styblinski–Tang
- Eggholder
- Six-Hump Camel
- Levy

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
