# COMBSS GLM

**COMBSS GLM** extends the original **COMBSS** framework to **generalized linear models (GLMs)**, providing a method for performing *best-subset selection* across non-linear likelihood-based models.

This repository builds on the methodology introduced in the original COMBSS project. In this implementation, we employ a Frank–Wolfe algorithm to handle box constraints, rather than unconstrained gradient descent. Currently, the algorithm supports logistic regression.

---

## Requirements

### Python
- **Python ≥ 3.10**

### Libraries
- `numpy >= 1.21.0`
- `scipy >= 1.12.0`  
- `scikit-learn >= 1.0.0`

---

## Example Usage

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import combssGLM.logistic_fw as lfw

# Generate sample data (Logistic Regression)
n_features = 50
n_informative = 10

X, y = make_classification(
    n_samples=1000,
    n_features=n_features,
    n_informative=n_informative,
    n_redundant=0,
    n_repeated=0,
    random_state=42,
    shuffle=False  # <--- IMPORTANT: keep informative features in 1..informative
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lfw.model()
model.fit(
    X_train=X_train,
    y_train=y_train,
    q=10,        # Maximum subset size
    max_iter=100
)
```

See `demo.py` for a more detailed example.

---

## Related Project

This work is an extension of **COMBSS**:  
➡️ https://github.com/saratmoka/COMBSS

---

## Authors

- **Dr. Anant Mathur** (UNSW Sydney)  
- **Dr. Sarat Moka** (UNSW Sydney)  
- **Prof. Benoît Liquet** (Macquarie University)
