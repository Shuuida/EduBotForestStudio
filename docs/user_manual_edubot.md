# EduBot – User Manual
**Version:** 1.0  
**Developed by:** EduBot Team  
**Author:** Michego Takoro  
**Date:** 2025  

---

## 🧩 Introduction

EduBot is an educational embedded machine learning platform designed to teach AI and robotics on low-cost hardware like Arduino or MegaPi.  

It allows users to **train ML models directly in Python**, export them into **C firmware**, and execute them on real microcontrollers for educational purposes.

---

## ⚙️ Main Features

- **Local ML Training** without NumPy or TensorFlow.  
- **C Firmware Export** for microcontrollers.  
- **Cross-Firmware Compatibility** using simple matrix multiplication.  
- **Interactive Visual Environment** for students and educators.

---

## 🧠 Supported Learning Models

| Model | Description | Best Use |
|--------|--------------|----------|
| DecisionTreeClassifier | Hierarchical condition-based classifier. | Binary detection tasks. |
| RandomForestClassifier | Ensemble of decision trees. | Noisy data and improved accuracy. |
| LinearRegression | Predictive regression model. | Continuous predictions. |
| MiniSVM | Lightweight Support Vector Machine. | Linear classification tasks. |
| MiniNeuralNetwork | Multilayer neural network with backpropagation. | Educational robotics and sensor learning. |

---


## 🧩 Example – Neural Network XOR

```python
from core import ml_runtime

model = ml_runtime.MiniNeuralNetwork(
    n_inputs=2,
    n_hidden=4,
    n_outputs=1,
    learning_rate=0.5,
    epochs=3000
)

X = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]

model.fit(X, y)
print("Predictions:", [model.predict(x) for x in X])
```

## ⚠️ Limitations
- Educational firmware only (limited RAM and float precision).

- Avoid networks with more than one hidden layer on microcontrollers.

- Dataset size should remain under 200 samples.

- Double precision (double) is not supported in firmware.



## 📜 License
EduBot ML Runtime © 2025 by Michego Takoro & EduBot Team.
For educational and research use only.
