# CSYE7105 HIGH PARALLEL MACHINE LEARNING & AI 
# Parallelized Payment Fraud Detection System

## Overview

This project is part of the CSYE7105 final submission by Team 8 (Hemant Gaikwad). It aims to build a scalable, parallelized system for detecting payment fraud in large financial datasets using Python, Dask, and XGBoost.

We utilize parallel computing frameworks to accelerate data preprocessing and model training across multiple CPU and GPU cores. The project demonstrates how to handle class imbalance, optimize feature engineering, and train machine learning models efficiently using both multiprocessing and distributed computing tools.

---

## Features

- ✅ One-hot encoding using Dask-ML's `DummyEncoder`
- ✅ Feature engineering with Dask DataFrames
- ✅ Handling extreme class imbalance through oversampling
- ✅ Scalable model training using XGBoost on CPU and GPU
- ✅ Parallel hyperparameter tuning with `GridSearchCV`
- ✅ Performance benchmarking across multiple cores

---

## Dataset

- **Source:** [Kaggle - Online Payment Fraud Detection](https://www.kaggle.com/datasets/fadyesam/online-fraud-detection)
- **Size:** ~482 MB
- **Records:** ~6.3 million
- **Target Column:** `isFraud`
- **Class Imbalance:** Ratio of ~774:1 (Non-fraud vs Fraud)

---

## Technologies Used

- Python 3.x
- Pandas & NumPy
- Dask (DataFrame & Array)
- Dask-ML
- Scikit-learn
- XGBoost (with GPU support)
- Multiprocessing
- Dask Distributed (optional for future extension)

---

## Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/parallel-fraud-detection.git
cd parallel-fraud-detection
````

2. **Install dependencies**

It is recommended to use a virtual environment or conda environment.

```bash
pip install -r requirements.txt
```

Example requirements:

```text
pandas
numpy
dask[dataframe]
dask-ml
scikit-learn
xgboost
```

3. **Download Dataset**

Place the dataset file `onlinefraud.csv` in the root directory of the project.

---

## How to Run

```bash
python main.py
```

> Make sure your system supports multiprocessing and GPU (for GPU runs). Modify `core_counts` and `use_gpu` flags inside the script as needed.

---

## Project Structure

```text
.
├── main.py                # Main script to run the pipeline
├── onlinefraud.csv        # Dataset file (you need to download this)
├── README.md              # Project description and usage
└── requirements.txt       # Python dependencies
```

---

## Key Learnings

* Dask enables lazy and parallel computation for big data pipelines
* Multiprocessing can be counterproductive if used without profiling
* GPU training with XGBoost significantly outperforms CPU on large datasets
* `.persist()` is crucial for caching intermediate Dask transformations to avoid recomputation
* Handling data imbalance is essential for effective fraud detection

---

## Contributor

* **Hemant Gaikwad**


---

## References

* Kaggle Dataset: [https://www.kaggle.com/code/fadyesam/online-fraud-eda-and-prediction-with-accuracy-99](https://www.kaggle.com/code/fadyesam/online-fraud-eda-and-prediction-with-accuracy-99)
* Dask Documentation: [https://docs.dask.org/](https://docs.dask.org/)
* XGBoost GPU Training: [https://xgboost.readthedocs.io/en/stable/gpu/index.html](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

---

## License

This project is for academic purposes only.


```
