# Delivery Delay Prediction & Explanation System

## Overview
This project predicts delivery delays and provides explanations for predictions using a combination of classical machine learning and large language models (LLMs). The system includes a full pipeline from data processing, feature engineering, model training, and fine-tuning, to a frontend dashboard for interactive exploration.  

The project workflow is demonstrated in the accompanying video.

## Demo Video
Watch the demo video here: [Demo Video](./demo_video.mp4)

---

## Project Workflow

### 1. Dataset
- Collected and curated delivery-related datasets, including order, customer, and shipment information.
- Dataset includes features like profit per order, item price, discounts, shipping mode, customer and order location, and payment type.

### 2. Exploratory Data Analysis (EDA) & Feature Engineering
- Conducted EDA to understand distributions, correlations, and patterns.
- Derived additional features:
  - Distance between store and order location.
  - Day-part features (morning, evening, night).
  - Order-to-shipment days and delays.
  - One-hot encoding for categorical variables (e.g., payment type, shipping mode).
- Computed performance scores for locations, days, and hours using historical data.

### 3. Model Training
- Trained a **Random Forest** classifier to predict delivery delays.
- Hyperparameter tuning using **Bayesian Optimization** to maximize model performance.
- Evaluated model using standard metrics (accuracy, F1-score, etc.).

### 4. Model Quantization
- Quantized the trained Random Forest model for faster inference and smaller size.
- Prepared a synthetic dataset for LLaMA 3.2 1B Instruct model fine-tuning.

### 5. LLaMA 3.2 1B Fine-Tuning
- Fine-tuned the model on **RunPod.io** using the synthetic dataset.
- Converted the model to **GGUF format** and quantized to 4-bit for efficient deployment.

### 6. Backend API
- Built a **FastAPI** backend:
  - Handles feature processing, model inference, and LLM explanations.
  - Generates SHAP explanations for Random Forest predictions.
  - Integrates LLaMA 3.2 1B for textual explanations.
- The backend was deployed on **AWS EC2** for demonstration purposes.

### 7. Frontend
- Developed an interactive **Streamlit** dashboard:
  - Collects input features.
  - Sends requests to the backend API.
  - Displays predictions and explanations from the Random Forest and LLM.

---

## Deployment
- Deployed the backend API on **AWS EC2**.
- Hosted the frontend dashboard on **Streamlit**.
- Recorded the full workflow in a demo video.
- EC2 and Streamlit instances were terminated post-demo to reduce costs.

---

## Future Scope
- Implement a **continual learning setup** to adapt the model to new data automatically.
- Incorporate **transformer-based architectures** for direct delivery delay prediction.
- Optimize the system for production deployment with robust monitoring and logging.
