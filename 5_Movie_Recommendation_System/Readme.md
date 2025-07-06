# 🎬 Movie Recommendation System

This project is a machine learning-based movie recommendation engine. It suggests similar movies to the one selected by the user based on content-based filtering techniques.

---

## 📁 Project Structure

.
├── screenshots/ # Folder containing screenshots of the web app
├── working video/ # Demo video showing the project in action
├── app.py # Streamlit app for movie recommendation
├── movies.csv # Dataset containing movie information
├── train_model.py # Script to build the similarity matrix
---

## 🧠 Technologies Used

- Python 🐍  
- Pandas 📊  
- Scikit-learn 🤖  
- Streamlit 🎛️  
- Cosine Similarity (Content-based Filtering)

---

## 📊 Dataset Overview

- **File**: `movies.csv`
- Columns may include:
  - `movieId`
  - `title`
  - `genres`
  - (additional metadata if added: `overview`, `cast`, etc.)

This dataset is used to generate a similarity matrix based on movie genres or descriptions.

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install pandas scikit-learn streamlit

2. Train the Recommendation Model
python train_model.py
This will generate a similarity matrix using TF-IDF or CountVectorizer + Cosine Similarity.

3. Launch the Streamlit Web App
streamlit run app.py
Then open http://localhost:8501 in your browser to interact with the recommender.

📷 Screenshots
UI screenshots are available in the screenshots/ folder.

🎥 Demo
A working demonstration video is available in the working video/ folder.

🙋‍♀️ Authors
Vaishnavi Borse
