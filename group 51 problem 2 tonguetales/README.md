# deeplearning_hackthon
 made for deeplearning  CS671 coursework  hackthon 

google drive link for complete project that have all the models and EVLUATIONS : https://drive.google.com/file/d/1QfW9ZXuvmgY8Kf93dPABIozf-f4MLQIW/view?usp=sharing

# TongueTales – AI-Powered Tongue Feature Extraction and Health Scoring

## 🧠 Overview

TongueTales is a computer vision and AI-based system that analyzes tongue images to extract five critical features traditionally used in Eastern medicine and computes two holistic health scores—Nutrition Score and Mantle Score. It also integrates a medical LLM (LLaMA3-Med42) for generating personalized health insights.

---

## 📦 Features Extracted (0–10 scale)

1. **Coated Tongue** – Degree of white/yellowish coating  
2. **Jagged Tongue Shape** – Serrated or uneven edges  
3. **Cracks on Tongue** – Fissures or lines  
4. **Filiform Papillae Size** – Fine hair-like projections  
5. **Redness of Fungiform Papillae** – Color intensity of red papillae

---

## ⚙️ System Pipeline

1. **Tongue Segmentation**  
   - Model: `U2NET` (fine-tuned on 2600+ images)  
   - Removes lips, teeth, and background.

2. **Preprocessing**  
   - `normalize_color()`: Adjusts for lighting.  
   - `standardize_orientation()`: Aligns tongue.

3. **Feature Extraction**  
   - `extract_coating_score()`  
   - `extract_jagged_shape_score()`  
   - `extract_cracks_score()`  
   - `extract_filiform_papillae_size_score()`  
   - `extract_fungiform_papillae_redness_score()`  

4. **Classification**  
   - Model: `ResNet152` (fine-tuned)  
   - Classes:  
     ['crenated tongue', 'crenated', 'fissured tongue', 'fissured', 'normal tongue', 'normal']

5. **Scoring Formulas**

nutrition_score = (
    0.4 * inverted_coating_score +
    0.3 * filiform_papillae_size_score +
    0.3 * fungiform_papillae_redness_score
)

mantle_score = (
    0.6 * inverted_cracks_score +
    0.4 * inverted_jagged_shape_score
)

6. **AI Analytics with LLaMA3-Med42**

Prompt passed to `llama3-med42` via Ollama:

"My tongue has a coating score (each out of 10) of X, Predicted class: Y, crack score of Z, ... Nutrition score: A, Mantle score: B. Give me an analysis in less than 30 words"

---

## 🛠️ Prerequisites

Install Python dependencies:

pip install flask torch torchvision numpy opencv-python pillow matplotlib flask-cors ollama

Download LLaMA3-Med42:

ollama pull llama3-med42

---

## 🚀 Run the Server

Ensure models are stored in:

checkpoints/u2net_finetuned_with2600_data_more.pth

Start the Flask server:

python app.py

---

## 🌐 API Endpoint

- POST /analyze  
  Accepts a tongue image and returns a detailed analysis.

### ✅ Response Format (JSON)

{
  "original_image": "...",
  "segmented_tongue": "...",
  "normalized_tongue": "...",
  "standardized_tongue": "...",
  "predicted_class": "normal tongue",
  "features": {
    "coating": {"score": 6.2, "visualization": "..."},
    ...
  },
  "scores": {"nutrition_score": 6.13, "mantle_score": 4.68},
  "ai_analysis": "Tongue shows moderate coating and cracks; consider hydration and iron intake..."
}

---

## 🧑‍💻 Frontend Integration (React)

This JSON response is consumed by a React-based frontend to display:
- Image visualizations
- Feature scores
- LLM analysis

---

## 📁 Folder Structure

.
├── main_server_flask_v1.0.1.py
├── checkpoints/
├── model
├──react_app

---

## ✅ Hackathon Submission Includes

- ✔️ Flask API  
- ✔️ Fine-tuned U2NET & ResNet152  
- ✔️ Custom feature extraction  
- ✔️ Scoring & visual output  
- ✔️ AI analysis using LLaMA3-Med42  
- ✔️ React-compatible output  

---

## ✨ Bonus Features

- Real-time webcam input (optional)  
- Heatmaps & overlays for explainability  
- Local inference with zero cloud dependency
