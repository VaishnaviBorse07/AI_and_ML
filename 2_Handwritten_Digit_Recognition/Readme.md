# ✍️ Handwritten Digit Recognition

This project is a deep learning-based web app that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. It provides an interactive interface built with Streamlit where users can draw digits and get predictions in real-time.

---

## 📁 Project Structure

.
├── screenshot/ # Folder containing screenshots of the project
├── working video/ # Folder containing demonstration video
├── app.py # Streamlit app to draw digits and predict
├── mnist_cnn_model.h5 # Trained CNN model for digit recognition
├── train_model.py # Script to train and save the CNN model

---

## 🧠 Technologies Used

- Python 🐍
- TensorFlow/Keras 🧠
- NumPy 📐
- Streamlit 🎛️

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd your-repo-directory
2. Install Dependencies

pip install -r requirements.txt
If requirements.txt is not available, install manually:

pip install tensorflow streamlit numpy
3. Train the Model (Optional)
If you want to retrain the model:

python train_model.py
This will generate:

mnist_cnn_model.h5

4. Run the Streamlit App

streamlit run app.py
Then open http://localhost:8501 in your browser.

🧾 Features
Draw digits using a canvas.

Predicts the digit using a trained CNN model.

Real-time inference using Keras .h5 model.

Simple and clean UI built with Streamlit.

🖼️ Screenshots
Find screenshots in the screenshot/ folder.

🎥 Demo
Watch the working demo video in the working video/ directory.

🙋‍♀️ Authors
Vaishnavi Borse
