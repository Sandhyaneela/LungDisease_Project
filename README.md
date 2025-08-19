# LungDisease_Project
🫁 Multi-Lung Disease Detection using X-ray Images

This project is a deep learning-based web application that detects multiple lung diseases from chest X-ray images. It helps in early screening and faster diagnosis by classifying X-rays into four categories:

Normal Lungs
COVID-19
Pneumonia
Tuberculosis (TB)
The system is built with Convolutional Neural Networks (CNNs) and deployed using a Flask web interface.

🚀 Features
✔️ Upload chest X-ray images for prediction
✔️ Model outputs disease type + confidence score
✔️ Generates a PDF report with results & suggested next steps
✔️ Easy-to-use Flask web app interface

🛠️ Tech Stack
Python (TensorFlow/Keras, OpenCV, NumPy, Matplotlib)
Flask (for web deployment)
ReportLab (for PDF generation)

📂 Project Structure
MultiLungDiseaseDetection/
│── models/             # Trained models
│── static/             # CSS, images
│── templates/          # HTML templates for Flask
│── uploads/            # Uploaded X-rays (runtime)
│── web_app.py          # Main Flask app
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation


⚙️ Installation & Setup

1️⃣ Clone this repository
git clone https://github.com/your-username/MultiLungDiseaseDetection.git
cd MultiLungDiseaseDetection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Flask app
python web_app.py

4️⃣ Open in browser
http://127.0.0.1:5000/


📊 Sample Output
Prediction: Pneumonia (92% confidence)
Heatmap highlighting infected lung regions
Auto-generated PDF report with next steps
(Add screenshots of your web app & reports here!)

✅ Future Enhancements
Expand dataset for higher accuracy
Support additional lung diseases
Using heatmaps
Deploy on cloud (AWS/Heroku/Streamlit) for easy access

