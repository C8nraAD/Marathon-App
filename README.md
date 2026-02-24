# Half-Marathon Calculator

A machine learning web application that predicts half-marathon finish times based on a brief conversational input. It utilizes a Ridge Regression model trained on historical race data to generate realistic pacing profiles and benchmark comparisons.

## Key Features
* **Conversational Input:** Collects user parameters (gender, age, 1km pace) via a chat interface with deterministic RegEx validation.
* **ML Inference:** Uses a Scikit-learn Pipeline (StandardScaler + Ridge Regression) to estimate total finish time.
* **Pacing Strategy:** Generates a 22-waypoint pacing profile derived from historical pace stability metrics.
* **Cloud Integration:** Fetches reference datasets and the serialized ML model directly from S3-compatible cloud storage.
* **LLM Integration:** Provides customized training tips using the OpenAI API.

## Tech Stack
* **Language:** Python 3.10+
* **Frontend:** Streamlit, Altair
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Cloud & External APIs:** Boto3 (S3), OpenAI API

## Model Performance
* **Dataset:** 18,377 runners (aggregated 2023-2024 data).
* **Metrics:** R-squared: 0.98 | MAE: ~54 seconds.
* **Note:** Ridge regression was selected to prevent overfitting and ensure stable predictions that account for standard pacing variance.

## Local Setup

1. Clone the repository:
```bash
git clone (https://github.com/C8nraAD/Half-Marathon-Calculator.git)
cd Half-Marathon-Calculator