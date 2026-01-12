# 📈 Tesla Stock Prediction Dashboard

A professional Streamlit dashboard for predicting Tesla stock prices using **LSTM neural networks**.  
Supports multiple **lookback windows (30, 60, 90, 120 days)** and prediction horizons (**1, 5, 10 days**).  
Interactive charts + predicted price tables with **light & dark themes**.

---

## 🚀 Features
- Dynamic **Lookback Window** selection
- Prediction horizons: **1, 5, 10 days**
- Interactive charts with **matplotlib**
- Professional **UI design** (light & dark mode)
- Real‑time training & forecasting
- Recruiter‑friendly demo mode

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.15-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b?logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-1.26-lightblue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.2-blue?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-green?logo=plotly)

---

## 📂 Project Structure

##tesla-stock-prediction/

│── app.py                 # Streamlit dashboard
│── data/TSLA.csv         # Tesla stock dataset
│── models/               # Saved models
│── requirements.txt       # Dependencies
│── README.md              # Documentation


---

## 📸 Screenshots

### Dashboard (Light Mode)
![Light Mode Screenshot](images/light_mode.png)

### Dashboard (Dark Mode)
![Dark Mode Screenshot](images/dark_mode.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/tesla-stock-prediction.git
cd tesla-stock-prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
Sayyed Mohsin Ali  
Intern Data Scientist 