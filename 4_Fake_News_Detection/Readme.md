# 📰 Fake News Detection

This project uses machine learning to detect whether a news article is real or fake based on its textual content. It helps combat misinformation by providing a tool to verify the authenticity of news headlines or articles.

## 📁 Project Structure

├── screenshot/ # Contains UI screenshots
├── working video/ # Folder with a demo video of the project
├── app.py # Streamlit app for user interaction and prediction
├── train_model.py # Python script to train and save the model


## 🧠 Technologies Used

- Python 🐍  
- Scikit-learn 🤖  
- Pandas 📊  
- TfidfVectorizer  
- Streamlit 🎛️ (for UI)

## 📊 Model Overview

- **Text Preprocessing**: Removing punctuation, stopwords, lowercase conversion
- **Vectorization**: TF-IDF Vectorizer
- **Model**: Logistic Regression / PassiveAggressiveClassifier (commonly used for fake news detection)


## 🚀 How to Run

### 1. Install Requirements

```bash
pip install pandas scikit-learn streamlit
2. Train the Model (Optional)
python train_model.py
This script processes the dataset, trains a model, and saves it as a .pkl file.

3. Run the Streamlit App
streamlit run app.py
Then open http://localhost:8501 in your browser.

🖼️ Screenshots
Visual previews of the UI are available in the screenshot/ folder.

🎥 Demo
View the working project in the working video/ folder.

🙋‍♀️ Authors
Vaishnavi Borse
