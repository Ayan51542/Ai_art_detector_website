# Step 1: Import the necessary tools
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline
from PIL import Image, ImageFilter
import io
import torch # PyTorch is needed for device management
import numpy as np
import cv2
import base64
import hashlib # For creating a unique hash of the image
import json # For handling data serialization
import os # To check for the service account key file

# --- Firebase/Firestore Integration ---
# This section initializes the connection to your database.
import firebase_admin
from firebase_admin import credentials, firestore

# Define the path to your service account key.
# This file should be in the same directory as your api.py script.
SERVICE_ACCOUNT_KEY_PATH = 'serviceAccountKey.json'
DB_STATUS = "Inactive"
db = None

try:
    # Check if the key file exists before trying to initialize
    if os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        # The project ID is automatically read from the key file.
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        DB_STATUS = "Active"
        print(f"Firestore initialized successfully. DB Status: {DB_STATUS}")
    else:
        print(f"!!! WARNING: Firestore key file not found at '{SERVICE_ACCOUNT_KEY_PATH}'. The app will run without database features. !!!")
        DB_STATUS = "Inactive - Key file not found"
except Exception as e:
    print(f"!!! FIRESTORE NOT INITIALIZED: {e}. The app will run without database features. !!!")
    DB_STATUS = f"Inactive - Error: {e}"


# --- HTML Template for the GUI ---
# Updated to include a gallery for forensic analysis images and DB status.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Multi-Model AI Art & Forensics Detector</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 40px auto; padding: 0 20px; background-color: #f8f9fa; }
        h1, h2 { color: #2c3e50; text-align: center; }
        .db-status { text-align: center; margin-bottom: 20px; padding: 10px; border-radius: 8px; font-weight: bold; }
        .db-active { background-color: #e8f5e9; color: #2e7d32; }
        .db-inactive { background-color: #ffebee; color: #c62828; }
        .container { background-color: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
        .main-layout { display: flex; flex-wrap: wrap; gap: 40px; }
        .column { flex: 1; min-width: 400px; }
        #image-preview-container { text-align: center; }
        #image-preview { max-width: 100%; max-height: 400px; border-radius: 8px; background-color: #ecf0f1; border: 2px dashed #bdc3c7; min-height: 200px; }
        input[type="file"] { border: 2px dashed #bdc3c7; padding: 20px; border-radius: 8px; width: 100%; box-sizing: border-box; background-color: #ecf0f1; margin-top: 10px; }
        button { background: linear-gradient(45deg, #8e44ad, #c0392b); color: white; padding: 14px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; width: 100%; margin-top: 20px; font-weight: bold; transition: transform 0.2s; }
        button:hover { transform: scale(1.02); }
        #results-container, #forensics-container { margin-top: 30px; display: none; }
        #spinner { display: none; margin: 20px auto; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #c0392b; border-radius: 50%; animation: spin 1s linear infinite; }
        #verdict { font-size: 24px; font-weight: bold; text-align: center; margin-bottom: 15px; }
        .real { color: #27ae60; }
        .fake { color: #c0392b; }
        .cached { color: #3498db; font-style: italic; font-size: 16px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        #forensics-gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; }
        .gallery-item img { width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd; }
        .gallery-item p { text-align: center; font-size: 12px; margin-top: 5px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced Multi-Model AI Art & Forensics Detector</h1>
        <div id="db-status-indicator" class="db-status">Database Connection: {{ db_status }}</div>
        <p style="text-align:center;">This tool uses a powerful ensemble of models and forensic analysis to provide a comprehensive result.</p>
        
        <div class="main-layout">
            <div class="column">
                <h2>Upload Image</h2>
                <div id="image-preview-container">
                    <img id="image-preview" src="" alt="Image preview will appear here">
                </div>
                <form id="upload-form">
                    <input type="file" id="image-input" name="image" accept="image/*" required>
                    <button type="submit">Analyze Image</button>
                </form>
            </div>
            <div class="column">
                <h2>Deep Learning Analysis</h2>
                <div id="spinner"></div>
                <div id="results-container">
                    <div id="verdict"></div>
                    <table id="results-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>AI Score</th>
                                <th>Real Score</th>
                                <th>Label</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div id="forensics-container">
                    <h2>Forensic Analysis</h2>
                    <div id="forensics-gallery"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('upload-form');
        const imageInput = document.getElementById('image-input');
        const imagePreview = document.getElementById('image-preview');
        const resultsContainer = document.getElementById('results-container');
        const forensicsContainer = document.getElementById('forensics-container');
        const tableBody = document.querySelector("#results-table tbody");
        const verdictDiv = document.getElementById('verdict');
        const forensicsGallery = document.getElementById('forensics-gallery');
        const spinner = document.getElementById('spinner');
        const dbStatusDiv = document.getElementById('db-status-indicator');

        // Set DB status indicator color
        if (dbStatusDiv.textContent.includes('Active')) {
            dbStatusDiv.classList.add('db-active');
        } else {
            dbStatusDiv.classList.add('db-inactive');
        }

        imageInput.addEventListener('change', () => {
            const file = imageInput.files[0];
            if (file) {
                imagePreview.src = URL.createObjectURL(file);
            }
        });

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (!imageInput.files.length) {
                alert('Please select an image file first.');
                return;
            }
            const formData = new FormData();
            formData.append('image', imageInput.files[0]);
            resultsContainer.style.display = 'none';
            forensicsContainer.style.display = 'none';
            tableBody.innerHTML = '';
            forensicsGallery.innerHTML = '';
            spinner.style.display = 'block';

            try {
                const response = await fetch('/detect', { method: 'POST', body: formData });
                if (!response.ok) {
                    let errorData;
                    try {
                        errorData = await response.json();
                    } catch (e) {
                        errorData = { error: response.statusText };
                    }
                    throw new Error(errorData.error || 'Server error');
                }
                const data = await response.json();
                displayResults(data);

            } catch (error) {
                displayError(error.message);
            } finally {
                spinner.style.display = 'none';
            }
        });

        function displayError(message) {
            resultsContainer.style.display = 'block';
            verdictDiv.innerHTML = `<span class="fake">Error</span>`;
            tableBody.innerHTML = `<tr><td colspan="4">${message || 'Unknown error'}</td></tr>`;
        }

        function displayResults(data) {
            resultsContainer.style.display = 'block';
            
            let verdictText = `${data.final_verdict} (Weighted Score: ${data.weighted_score.toFixed(2)}%)`;
            if (data.cached) {
                verdictText += ` <span class="cached">[Cached Result]</span>`;
            }
            verdictDiv.innerHTML = verdictText;
            verdictDiv.className = data.final_verdict.includes('AI') ? 'fake' : 'real';

            // The JSON strings need to be parsed if they exist (for cached results)
            let individual_results = data.individual_results_json ? JSON.parse(data.individual_results_json) : data.individual_results;
            
            // For cached results, we need to generate forensic images on the fly
            // For new results, they are sent directly.
            let forensic_images = data.forensic_images;

            if (individual_results) {
                individual_results.forEach(result => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${result.model_name}</td>
                        <td>${result.confidence.fake.toFixed(2)}%</td>
                        <td>${result.confidence.real.toFixed(2)}%</td>
                        <td><b>${result.prediction}</b></td>
                    `;
                    tableBody.appendChild(row);
                });
            }

            if (forensic_images) {
                forensicsContainer.style.display = 'block';
                forensic_images.forEach(item => {
                    const galleryItem = document.createElement('div');
                    galleryItem.className = 'gallery-item';
                    galleryItem.innerHTML = `
                        <img src="data:image/jpeg;base64,${item.image}" alt="${item.label}">
                        <p>${item.label}</p>
                    `;
                    forensicsGallery.appendChild(galleryItem);
                });
            }
        }
    </script>
</body>
</html>
"""

# --- Forensic & Utility Functions (from user's files) ---
def gen_ela(img_array, quality=90):
    """Generates an Error Level Analysis image."""
    if img_array.shape[2] == 4: # Handle RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', img_array, encode_param)
    compressed_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    ela_img = cv2.absdiff(img_array, compressed_img)
    ela_img = cv2.convertScaleAbs(ela_img, alpha=10)
    return Image.fromarray(cv2.cvtColor(ela_img, cv2.COLOR_BGR2RGB))

def gradient_processing(image_array):
    """Performs gradient processing to highlight edges."""
    gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(dx, dy)
    gradient_img = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(gradient_img)

def pil_to_base64(image):
    """Converts a PIL Image to a base64 string for embedding in HTML."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Flask App Setup ---
app = Flask(__name__)
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

MODELS_CONFIG = {
    "SwinV2 Based": {"path": "haywoodsloan/ai-image-detector-deploy", "weight": 0.15},
    "ViT Based": {"path": "Heem2/AI-vs-Real-Image-Detection", "weight": 0.15},
    "SDXL Dataset": {"path": "Organika/sdxl-detector", "weight": 0.15},
    "SDXL + FLUX": {"path": "cmckinle/sdxl-flux-detector_v1.1", "weight": 0.15},
    "DeepFake v2": {"path": "prithivMLmods/Deep-Fake-Detector-v2-Model", "weight": 0.15},
    "Midjourney/SDXL": {"path": "ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL", "weight": 0.10},
    "ViT v4": {"path": "date3k2/vit-real-fake-classification-v4", "weight": 0.15}
}

print("Loading ensemble of AI detection models... This may take several minutes.")
models = {}
for name, config in MODELS_CONFIG.items():
    print(f"--> Loading model: {name} ({config['path']})")
    try:
        models[name] = pipeline("image-classification", model=config['path'], device=device)
        print(f"--> {name} loaded successfully.")
    except Exception as e:
        print(f"--> FAILED to load {name}. This model will be skipped. Error: {e}")
print(f"Loaded {len(models)} out of {len(MODELS_CONFIG)} models. The server is ready.")

@app.route("/", methods=["GET"])
def home():
    # Pass the database status to the template
    return render_template_string(HTML_TEMPLATE, db_status=DB_STATUS)

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read() # Read bytes for hashing
    
    # Create a unique hash for the image to check for duplicates
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    
    # --- Database Check ---
    if db:
        print(f"DATABASE IS ACTIVE. Checking for hash: {image_hash[:10]}...")
        doc_ref = db.collection('image_analysis_cache').document(image_hash)
        doc = doc_ref.get()
        if doc.exists:
            print(f"CACHE HIT: Found result for image hash {image_hash[:10]} in Firestore.")
            cached_data = doc.to_dict()
            cached_data['cached'] = True # Add flag for frontend
            
            # Since we don't store forensic images, we generate them on the fly for cached results
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(input_image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            ela_img = gen_ela(img_bgr)
            gradient_img = gradient_processing(img_bgr)
            cached_data['forensic_images'] = [
                {"label": "Original", "image": pil_to_base64(input_image)},
                {"label": "ELA", "image": pil_to_base64(ela_img)},
                {"label": "Gradient", "image": pil_to_base64(gradient_img)}
            ]
            return jsonify(cached_data)
        else:
            print(f"CACHE MISS: No result found for hash {image_hash[:10]}.")
    else:
        print("DATABASE IS INACTIVE. Skipping cache check.")
    
    # --- If not in DB, proceed with full analysis ---
    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # --- Forensic Analysis ---
        print("Performing forensic analysis...")
        img_np = np.array(input_image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        ela_img = gen_ela(img_bgr)
        gradient_img = gradient_processing(img_bgr)
        
        forensic_images_b64 = [
            {"label": "Original", "image": pil_to_base64(input_image)},
            {"label": "ELA", "image": pil_to_base64(ela_img)},
            {"label": "Gradient", "image": pil_to_base64(gradient_img)}
        ]

        # --- Deep Learning Analysis ---
        print("Running weighted ensemble analysis...")
        individual_results = []
        weighted_ai_score = 0
        total_weight = 0

        for name, model_pipeline in models.items():
            model_weight = MODELS_CONFIG[name]["weight"]
            predictions = model_pipeline(input_image)
            confidence = {p['label'].lower(): p['score'] for p in predictions}
            
            artificial_score = (
                confidence.get('artificial', 0) or confidence.get('ai image', 0) or 
                confidence.get('ai', 0) or confidence.get('deepfake', 0) or 
                confidence.get('ai_gen', 0) or confidence.get('fake', 0)
            )
            real_score = (
                confidence.get('real', 0) or confidence.get('real image', 0) or 
                confidence.get('human', 0) or confidence.get('realism', 0)
            )
            
            if artificial_score > 0 and real_score == 0: real_score = 1.0 - artificial_score
            elif real_score > 0 and artificial_score == 0: artificial_score = 1.0 - real_score

            weighted_ai_score += artificial_score * model_weight
            total_weight += model_weight
            
            individual_results.append({
                "model_name": name,
                "prediction": "AI" if artificial_score > real_score else "REAL",
                "confidence": {"fake": artificial_score * 100, "real": real_score * 100}
            })

        final_weighted_score_percent = (weighted_ai_score / total_weight) * 100 if total_weight > 0 else 0
        final_verdict = "Consensus: Likely AI-Generated" if final_weighted_score_percent > 50 else "Consensus: Likely Human-Made (Real)"
        tag = "AI" if final_weighted_score_percent > 50 else "REAL"

        response_data = {
            "final_verdict": final_verdict,
            "weighted_score": final_weighted_score_percent,
            "individual_results": individual_results,
            "forensic_images": forensic_images_b64,
            "tag": tag
        }

        # --- Save to Database ---
        if db:
            print(f"SAVING to Firestore: Saving new result for image hash {image_hash[:10]}")
            # We ONLY store the results, not the images, to avoid exceeding size limits.
            db_data = {
                "final_verdict": response_data["final_verdict"],
                "weighted_score": response_data["weighted_score"],
                "individual_results_json": json.dumps(response_data["individual_results"]),
                "tag": response_data["tag"]
            }
            doc_ref.set(db_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    if models:
        app.run(host="0.0.0.0", port=5000)
    else:
        print("Application will not start because no models could be loaded.")
