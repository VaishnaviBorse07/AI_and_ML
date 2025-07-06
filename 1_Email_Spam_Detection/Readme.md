# 📧 Email Spam Detection

This project is a machine learning-based spam email classifier that detects whether a given email message is spam or not. It uses a TF-IDF vectorizer and a trained machine learning model (like Naive Bayes) to perform classification.

## 📁 Project Structure

.
├── screenshot/ # Folder containing screenshots of the project
├── working video/ # Folder containing demo or working video
├── app.py # Streamlit app for email spam detection
├── spam.csv # Dataset used to train the model
├── spam_classifier_model.pkl # Trained spam classifier model
├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
├── train_model.py # Script to train and save the model

---

## 🧠 Technologies Used

- Python 🐍
- Scikit-learn 🤖
- Pandas 📊
- Streamlit 🎛️
- Pickle 🥒

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd your-repo-directory
2. Install Dependencies

pip install -r requirements.txt
If requirements.txt is not available, install manually:

pip install pandas scikit-learn streamlit

3. Train the Model (optional)
python train_model.py
This will generate:

spam_classifier_model.pkl

tfidf_vectorizer.pkl

4. Run the Streamlit App
streamlit run app.py
Then, open http://localhost:8501 in your browser to access the app.

📊 Dataset
File: spam.csv

Contains email text and labels (spam or ham)

Used for training the spam detection model

📷 Screenshots
You can find UI previews in the screenshot/ directory.

🎥 Demo
A demo video is available in the working video/ directory.

🙋‍♀️ Authors
Vaishnavi Borse
