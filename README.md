# Advanced Machine Learning Project

This project explores **advanced supervised machine learning techniques**, focusing on
model training, evaluation, and comparison using real data.

The goal is to analyse model performance and understand the impact of different
learning approaches, hyperparameters, and evaluation metrics.

## Context
Academic project developed for the course **Aprendizagem Automática II (Machine Learning II)**.

## Objectives
- Train and evaluate multiple machine learning models
- Compare performance using appropriate metrics
- Analyse strengths and limitations of each approach

## Tech Stack
- Python
- NumPy, Pandas
- scikit-learn
- (Optional) TensorFlow / PyTorch
- Matplotlib / Seaborn

## Project Structure
- `src/train.py` — model training pipeline
- `src/evaluate.py` — evaluation and metrics
- `src/utils.py` — helper functions
- `data/` — dataset (if included)
- `models/` — saved trained models (if included)

## Evaluation
Models are evaluated using appropriate metrics such as:
- Accuracy / Precision / Recall / F1-score
- RMSE / MAE (for regression)
- Cross-validation (when applicable)

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
