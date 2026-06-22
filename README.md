# AI Fraud Detection

A simple fraud detection API built with Flask and scikit-learn.

## Project Structure

- `app.py` - Flask application that loads a saved model and exposes a `/predict` endpoint.
- `train_model.py` - Script for training a logistic regression model on the dataset and saving it as `fraud_model.pkl`.
- `requirements.txt` - Python dependencies.
- `dataset/spam.csv` - Example dataset containing text messages and labels.
- `fraud_model.pkl` - Trained model artifact.

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the environment:

- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
- Windows CMD:
```cmd
.\.venv\Scripts\activate.bat
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script to build the model and save it to `fraud_model.pkl`:

```bash
python train_model.py
```

## Run the API

Start the Flask app:

```bash
python app.py
```

The API will be available at `http://0.0.0.0:5000`.

## Predict

Send a POST request to `/predict` with JSON payload containing `text` or `message`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a fraudulent message"}'
```

Example response:

```json
{
  "prediction": "fraud",
  "fraud_probability": 0.8723
}
```

## Notes

- The model expects dataset rows with `text` and `label` columns.
- Labels should be `legit` or `fraud`.
- If you update the dataset, retrain the model before running predictions.
