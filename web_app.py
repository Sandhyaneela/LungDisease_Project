import os
import sys
import uuid
from flask import Flask, render_template, request, send_file, url_for
from reportlab.pdfgen import canvas
from Classes.Web_Model import predict_image

# Ensure the root project directory is in the system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'runs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Route: Home page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Get prediction and confidence
    predicted_class, confidence = predict_image(image_path)

    # Suggested next steps
    next_steps = {
        "NORMAL": "No disease detected. Maintain regular health checkups.",
        "TUBERCULOSIS": "Consult a pulmonologist. Begin anti-tuberculosis treatment.",
        "COVID19": "Self-isolate and consult a physician. Get RT-PCR confirmation.",
        "PNEUMONIA": "Seek immediate medical care. Antibiotics may be required."
    }
    suggestion = next_steps.get(predicted_class, "Please consult a doctor.")

    # Generate PDF report
    report_name = f"report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = os.path.join(REPORT_FOLDER, report_name)
    generate_pdf(report_path, file.filename, predicted_class, confidence, suggestion)

    return render_template(
        'result.html',
        prediction=predicted_class,
        confidence=confidence,
        suggestion=suggestion,
        pdf_filename=report_name
    )

# Route: Download PDF report
@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    file_path = os.path.join(REPORT_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_file(file_path, as_attachment=True)

# PDF generation function
def generate_pdf(path, image_name, prediction, confidence, suggestion):
    c = canvas.Canvas(path)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 800, "Lung Disease Detection Report")

    c.setFont("Helvetica", 14)
    c.drawString(100, 760, f"Image: {image_name}")
    c.drawString(100, 730, f"Prediction: {prediction}")
    c.drawString(100, 700, f"Confidence: {confidence:.2f}%")
    c.drawString(100, 670, f"Next Steps: {suggestion}")
    c.save()

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
