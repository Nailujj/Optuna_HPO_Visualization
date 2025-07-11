# Optuna_HPO_Visualization

**Visual comparison of Bayesian Hyperparameter Optimization and Grid Search in a controlled 3D objective space**

**Video Demonstration**: [https://youtu.be/PxGvPgyvNHE](https://youtu.be/PxGvPgyvNHE)

---

## Overview

This short animation visualizes the process of hyperparameter optimization using two approaches:

1. **Grid Search** – an exhaustive, deterministic method.
2. **Bayesian Optimization** – using the Tree-structured Parzen Estimator (TPE) as implemented in [Optuna](https://optuna.org).

The goal is to provide an intuitive, visual understanding of how Bayesian optimization methods more efficiently explore high-dimensional, non-convex objective spaces compared to naive grid-based sampling.

---

## Methodology

We define a synthetic two-dimensional continuous hyperparameter space:

- **Parameter 1**: Learning Rate ![lr](https://latex.codecogs.com/png.image?\dpi{120}x\in[10^{-6},10^{-4}]) (log-scale)  
- **Parameter 2**: Layer-wise LR Decay ![decay](https://latex.codecogs.com/png.image?\dpi{120}x\in[0.6,1.0])


The objective function *f(θ)* maps a 2D parameter vector *θ = (x, y)* to a scalar -F_1 value, and is defined as:

$$
f(x, y) = 1 - \left( 
    \frac{10 \cdot e^{-((x + 2)^2 + (y + 2)^2)}}{1.0} +
    \frac{14.5 \cdot e^{-((x)^2 + (y - 3)^2)}}{2.0} +
    \frac{5.0 \cdot e^{-((x - 3)^2 + (y + 1)^2)}}{1.5}
\right) \bigg/ \left(2.0 + 1.5 + 1.0\right)
$$

This function is multimodal and normalized to map values roughly into the range \([0, 1]\), with multiple local minima.

Two optimization strategies are then simulated:

### 1. Grid Search

- Uniform sampling across the parameter space.
- All samples are predetermined and independent of previous evaluations.
- Computationally inefficient in higher dimensions.

### 2. Optuna (TPE Sampler)

- Sequential model-based optimization using the [Tree-structured Parzen Estimator (TPE)](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html).
- Each trial is informed by prior evaluations.
- Focuses sampling on regions with higher expected improvement.

Both approaches are constrained to the same number of function evaluations to allow a fair comparison.


## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization*, Advances in Neural Information Processing Systems.
- [Manim](https://www.manim.community/): Animation engine for explanatory math videos.

---

## License

This project is licensed under the [MIT License](LICENSE).
