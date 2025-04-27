# 🩺 Pacemaker Detection in ECG Signals – Deep Learning Project

This project focuses on building a deep learning model to detect the presence of pacemakers in ECG signals. The work was carried out in collaboration with the AI Center for Cardiology at Sheba Medical Center.

The project will be presented as a poster at the **Israel Cardiology Association Conference** in **May 2025**.

---

## 🎯 Project Goals

- Develop an automated AI-based system to detect pacemakers from ECG data.
- Support research infrastructure at the Sheba Heart Research Center.
- Adapt public datasets for specialized medical use.
- Handle highly imbalanced data distributions.
- Train, validate, and evaluate models on real-world medical data.

---

## 🧠 Approach

- Conducted an extensive **literature review** to identify the best modeling strategies.
- Preprocessed ECG datasets (PhysioNet, PTB-XL) and integrated private Sheba data.
- Engineered wide features and used supervised learning.
- Applied **under-sampling** and **over-sampling** techniques to manage class imbalance.
- Designed and trained deep learning classification models using PyTorch.
- Evaluated models with **ROC-AUC**, **Confusion Matrices**, and **Classification Reports**.

---

## ⚙️ Technologies Used

- Python
- PyTorch
- Scikit-learn
- Numpy, Pandas
- WFDB, Biosppy (ECG signal processing libraries)
- Jupyter Notebook, VS Code
- Docker (for environment reproducibility)

---

## 🚀 How to Run

1. Clone the repository.
2. Install Python dependencies listed in `requirements.txt`.
3. Preprocess ECG data using provided scripts.
4. Train models via `train_model.py` or Jupyter Notebooks.
5. Evaluate performance using the evaluation scripts.

---

## 📊 Data Sources

- PhysioNet 2020 Challenge datasets
- PTB-XL ECG Database
- Private annotated data from Sheba Medical Center

---

## 📄 License

This project is intended for research purposes within the Sheba Medical Center. External use requires appropriate permissions.

---

## 👩‍💻 Developed by

Dvora Rothman, Elisheva Tufik, Sigal Sina, Gal Goshen, Raizy Kellerman, Michal Cohen-Shelly, Avi Sabbag
