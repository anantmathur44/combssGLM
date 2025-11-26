# COMBSS GLM

**COMBSS GLM** extends the original **COMBSS** framework to **generalized linear models (GLMs)**, providing tools for performing *combinatorial best-subset selection* across non-linear likelihood-based models.

This repository builds on the methodology introduced in the original COMBSS project. In this implementation, we employ a Frank–Wolfe algorithm to handle box constraints, rather than unconstrained gradient descent. Currently, the algorithm supports logistic regression.

---

## Requirements

### Python
- **Python ≥ 3.10**

### Core scientific computing libraries
- `numpy >= 1.21.0`
- `scipy >= 1.12.0`  

### Machine learning library
- `scikit-learn >= 1.0.0`

---

## Related Project

This work is an extension of **COMBSS**:  
➡️ https://github.com/saratmoka/COMBSS

---

## Authors

- **Dr. Anant Mathur** (UNSW Sydney)  
- **Dr. Sarat Moka** (UNSW Sydney)  
- **Prof. Benoît Liquet** (Macquarie University)
