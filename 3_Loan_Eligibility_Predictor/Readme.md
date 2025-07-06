# 🏦 Loan Eligibility Predictor

This project predicts whether an applicant is eligible for a loan based on input features like income, education, marital status, credit history, etc. It uses machine learning techniques trained on real-world-like loan data.

---

## 📁 Project Structure

.
├── screenshot/ # Folder containing UI screenshots
├── working video/ # Demo video of the application
├── eligiblity_predictor.py # Python script with the model and prediction logic
├── train.csv # Training dataset
├── test.csv # Testing/validation dataset

---

## 🧠 Technologies Used

- Python 🐍
- Pandas 📊
- Scikit-learn 🤖
- Streamlit (optional for UI)
- Jupyter (optional for training/EDA)

---

## 📊 Dataset Overview

- **train.csv**: Used to train the ML model.
- **test.csv**: Used to test the model’s prediction accuracy.
- Features include:
  - `Gender`
  - `Married`
  - `Dependents`
  - `Education`
  - `Self_Employed`
  - `ApplicantIncome`
  - `CoapplicantIncome`
  - `LoanAmount`
  - `Loan_Amount_Term`
  - `Credit_History`
  - `Property_Area`
  - `Loan_Status` (target)

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install pandas scikit-learn

If you are using a Streamlit interface, also run:
pip install streamlit

2. Run the Predictor Script
python eligiblity_predictor.py

If app.py or a Streamlit version is used:
streamlit run eligiblity_predictor.py

🖼️ Screenshots
You can find demo screenshots in the screenshot/ folder.

🎥 Demo Video
Check out the project workflow in the working video/ folder.

🙋‍♀️ Authors
Vaishnavi Borse
