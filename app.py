
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import csv
from datetime import datetime
from fastapi.staticfiles import StaticFiles
import json



app = FastAPI(title="Tomato Disease Diagnostic API")

from fastapi.middleware.cors import CORSMiddleware

# This is the "Magic Key" that lets your website talk to your AI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later we can put your specific Vercel link here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Server is Running!"}



# --- STEP 2: OPEN THE WINDOW ---
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- NEW: LOAD THE ENCYCLOPEDIA ---
TREATMENT_JSON_PATH = "treatments.json"

if os.path.exists(TREATMENT_JSON_PATH):
    with open(TREATMENT_JSON_PATH, "r", encoding="utf-8") as f:
        TREATMENT_DB = json.load(f)
else:
    print("⚠️ WARNING: treatments.json not found! Using empty data.")
    TREATMENT_DB = {}

# Allow connections from mobile/web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. THE BRAIN (Architecture) ---

class MetaACON(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        beta = self.fc(x)
        return (self.p1 - self.p2) * x * torch.sigmoid(beta * (self.p1 - self.p2) * x) + self.p2 * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(res))

class GatekeeperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b3(weights=None)
        self.base.classifier[1] = nn.Linear(1536, 3)
    def forward(self, x): return self.base(x)

class DiseaseModel(nn.Module):
    def __init__(self, NUM_CLASSES=6):
        super().__init__()
        self.base = models.efficientnet_v2_s(weights=None)
        self.features = self.base.features
        self.attention = SpatialAttention()
        self.activation = MetaACON(1280)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1280, NUM_CLASSES)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.activation(x)
        return self.classifier(x)

# --- 2. LOADING ---

# --- HOW TO LOAD THEM SAFELY ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Gatekeeper
gatekeeper = GatekeeperModel().to(device)
gatekeeper.base.load_state_dict(torch.load('model/stage2_bestttTTT211_model.pth', map_location=device))
gatekeeper.eval() # IMPORTANT for prediction

GATEKEEPER_CLASSES = [
    "Non___Leaf",
    "Other___Plant_Leaf",
    "Tomato___Leaf"
]

# Load Disease Doctor
disease_doctor = DiseaseModel(NUM_CLASSES=6).to(device)
disease_doctor.load_state_dict(torch.load('model/stage3_best_model22.pth', map_location=device))
disease_doctor.eval() # IMPORTANT for prediction

DISEASE_CLASSES = [
    "Other___Tomato_disease",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 3. ENDPOINTS ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    content = await file.read()

     # 2. GIVE THE PHOTO A NAME (Using the current time)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{timestamp}_{file.filename}"
    
    # 3. PUT THE PHOTO IN THE ALBUM (The folder you created)
    # image_path = f"/static/uploaded_images/{image_name}"
    # with open(image_path, "wb") as f:
    #     f.write(content)

    upload_folder = "static/uploaded_images"
    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, image_name).replace("\\", "/")

    with open(image_path, "wb") as f:
        f.write(content)

    image = Image.open(io.BytesIO(content)).convert('RGB')
    display_img = image.resize((800, 800), Image.Resampling.LANCZOS)
    display_img.save(image_path, quality=95) # High-quality save
    x = transform(image).unsqueeze(0).to(device)

    # -------- STAGE 1: THE RE-CALIBRATED GATEKEEPER --------
    with torch.no_grad():
      gate_out = gatekeeper(x)
      gate_out[0][2] += 0.8  # Keep your gentle bias
      gate_prob = F.softmax(gate_out / 2.2, dim=1)

      gate_conf, gate_pred = torch.max(gate_prob, 1)
      gate_label = GATEKEEPER_CLASSES[gate_pred.item()]

    tomato_prob = gate_prob[0][2].item()

    # NEW LOGIC: If Tomato confidence is too low, reject it immediately
    if tomato_prob < 0.48:  # I increased this slightly to 0.48 to catch that 0.44 Aloe Vera
            if gate_pred == 0:
                msg = "Wait! We can't find a leaf here. Please point the camera at a single Tomato leaf."
                error_type = "NON_LEAF"
            elif gate_pred == 1:
                msg = "This looks like a different plant's leaf! Our AI only specializes in Tomato leaves."
                error_type = "WRONG_PLANT"
            else:
                msg = "The image is not clear enough to identify as a tomato leaf."
                error_type = "UNCLEAR"

            # This sends the specific message to Lovable
            return {
            "status": "rejected",
            "error_type": error_type,
            "message": msg
        }

    # If it passes 0.48, then it is a "Confirmed Tomato"
    else:
            print(f"✅ Tomato Leaf identified (Confidence: {tomato_prob:.2f}). Analyzing health...")

        # -------- STAGE 2: STABILIZED DISEASE DOCTOR --------
    with torch.no_grad():
            dis_out = disease_doctor(x)
            # T=3.0 fixes the "arrogant" high confidence lies of the 89% model
            T = 3.0
            prob1 = F.softmax(dis_out / T, dim=1)

            top_probs, top_idxs = torch.topk(prob1, 2)
            conf1 = top_probs[0][0].item()
            conf2 = top_probs[0][1].item()
            gap = conf1 - conf2

            label1 = DISEASE_CLASSES[top_idxs[0][0].item()]
            label2 = DISEASE_CLASSES[top_idxs[0][1].item()]
        # -------- STAGE 3: THE "DEEP INSPECTION" LOGIC --------

    # Calculate the sum of all disease probabilities (excluding 'Healthy')
    # This acts as a "Disease Radar" for distant images.
    healthy_idx = DISEASE_CLASSES.index('Tomato___healthy')
    all_probs = prob1[0].tolist()
    disease_sum = sum([p for i, p in enumerate(all_probs) if i != healthy_idx])

    # CASE: CAUTIOUS HEALTHY (The Polite Warning)
    if label1 == 'Tomato___healthy' and (disease_sum > 0.30 or conf1 < 0.60):
        result_status = "distant_view"
        message = "The leaf looks mostly healthy, but the AI sees suspicious spots starting to form. Please take a closer, well-lit photo of those specific spots for a deeper check!"
        treatment_data = TREATMENT_DB.get(label1, {})
        treatment2 = None

    # CASE: AMBIGUOUS (The Dual Diagnosis / Side-by-Side)
    elif gap < 0.60:
        result_status = "ambiguous"
        message = f"Symptoms resemble both {label1} and {label2}. Compare the samples below."
        treatment_data = TREATMENT_DB.get(label1, {})
        treatment2 = TREATMENT_DB.get(label2, {})

    # CASE: CERTAIN (Includes 'Other' and 'Success' results)
    else:
        result_status = "success"
        message = f"Diagnosis confirmed: {label1}"
        treatment_data = TREATMENT_DB.get(label1, {})
        treatment2 = None

    # 6. FINAL RESPONSE TO LOVABLE
    # 6. FINAL RESPONSE TO LOVABLE (The "Visual Match" Concept)
    
    # --- PREPARE THE DATA FOR LOVABLE & SUPABASE (English Only) ---
    now = datetime.now()
    
    # This is the "Sticky Note" Lovable will use for the History Table

    # Use the 'Forwarding' URL from your terminal screen
    MY_PUBLIC_URL = "https://tinglingly-runtgenological-earl.ngrok-free.dev"
    clean_path = image_path.lstrip("/")
    full_image_url = f"{MY_PUBLIC_URL}/{clean_path}"
    history_record = {
        "disease_name": label1,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        # "image_url": f"{MY_PUBLIC_URL}/{image_path}", 
        "image_url": full_image_url,
        "is_healthy": label1 == "Tomato___healthy"
    }




    # # Before the images and treatment chnages 
    # # --- STEP 4: DYNAMIC FOLDER SCANNER ---
    # def get_choice_data(label):
    #     """Prepares the folder path and library data for the frontend."""
    #     base_ref_path = "static/reference"
        
    #     # Check if this is the "Other" category to build the Library
    #     if label == "Other___Tomato_disease":
    #         other_path = os.path.join(base_ref_path, "Other___Tomato_disease")
    #         # Scan all sub-folders (Mosaic, Septoria, etc.)
    #         try:
    #             sub_folders = [f for f in os.listdir(other_path) if os.path.isdir(os.path.join(other_path, f))]
    #         except Exception:
    #             sub_folders = []
            
    #         return {
    #             "label": label,
    #             "display_name": "🔍 Other Tomato Diseases",
    #             "is_library": True,
    #             "library_data": [
    #                 {
    #                     "name": f.replace("_", " "),
    #                     "folder_path": f"static/reference/Other___Tomato_disease/{f}",
    #                     "treatment":TREATMENT_DB.get(f, {"message": "No specific treatment found for this variant."})
    #                 } for f in sub_folders
    #             ],
    #             "details": TREATMENT_DB.get(label, {})
    #         }
    #     else:
    #         # Standard disease logic
    #         return {
    #             "label": label,
    #             "display_name": label.replace("Tomato___", "").replace("_", " "),
    #             "is_library": False,
    #             "image_folder": f"static/reference/{label}",
    #             "details": TREATMENT_DB.get(label, {})
    #         }

    # # --- STEP 5: FINAL RESPONSE TO LOVABLE ---
    # return {
    #     "status": result_status,
    #     "message": message,
    #     "user_image": f"/{image_path}", # Path to the photo the user just took
    #     "history_data": history_record,
        
    #     "choice1": get_choice_data(label1),
        
    #     "choice2": get_choice_data(label2) if result_status == "ambiguous" else None
    # }



    

    # After the images and treatment chnages...
    # # --- STEP 4: DYNAMIC FOLDER SCANNER (UPDATED FOR 3 IMAGES) ---
    # def get_choice_data(label):
    #     """Finds all 3 images and prepares structured treatment data."""
    #     raw_data = TREATMENT_DB.get(label, {})
        
    #     # Path for the browser/frontend (Must start with /static/)
    #     browser_base = f"/static/reference/{label}"
        
    #     # Path for your laptop to scan files
    #     full_os_path = os.path.join("static", "reference", label)
        
    #     # 1. SCAN FOR ALL 3 IMAGES (1.jpg, 2.jpg, 3.jpg)
    #     image_list = []
    #     if os.path.exists(full_os_path):
    #         try:
    #             # This finds every image in the folder and sorts them
    #             files = [f for f in os.listdir(full_os_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    #             files.sort()
    #             # Create the full Web URLs
    #             image_list = [f"{browser_base}/{img}" for img in files]
    #         except Exception:
    #             image_list = []

    #     if label == "Other___Tomato_disease":
    #         sub_folders = []
    #         try:
    #             sub_folders = [f for f in os.listdir(full_os_path) if os.path.isdir(os.path.join(full_os_path, f))]
    #         except Exception:
    #             sub_folders = []
            
    #         return {
    #             "label": label,
    #             "display_name": "🔍 Other Tomato Diseases",
    #             "is_library": True,
    #             "library_data": [
    #                 {
    #                     "name": f.replace("_", " "),
    #                     "folder_path": f"/static/reference/Other___Tomato_disease/{f}",
    #                     "treatment": raw_data.get(f, {})
    #                 } for f in sub_folders
    #             ],
    #             "details": raw_data
    #         }
    #     else:
    #         return {
    #             "label": label,
    #             "display_name": label.replace("Tomato___", "").replace("_", " "),
    #             "is_library": False,
    #             "image_list": image_list[:3], # This sends the 1.jpg, 2.jpg, 3.jpg list!
    #             "details": raw_data
    #         }
    
    #     # --- STEP 5: FINAL RESPONSE TO LOVABLE ---
    #     return {
    #     "status": result_status,
    #     "message": message,
    #     "user_image": f"/{image_path}", 
    #     "history_data": history_record,
    #     "choice1": get_choice_data(label1),
    #     "choice2": get_choice_data(label2) if result_status == "ambiguous" else None,
    #     # This adds the safety protocol for the popups
    #     "safety_protocol": TREATMENT_DB.get("Human_Safety_Protocol", {})
    #    }





    # --- STEP 4: DYNAMIC FOLDER SCANNER (WITH LIBRARY SLIDERS) ---
    def get_choice_data(label):
        """Finds images and prepares data, including sliders for the Library."""
        raw_data = TREATMENT_DB.get(label, {})
        full_os_path = os.path.join("static", "reference", label)
        browser_base = f"/static/reference/{label}"
        
        # 1. Standard Slider Scanner (for Main Results like Early Blight)
        image_list = []
        if os.path.exists(full_os_path):
            try:
                files = [f for f in os.listdir(full_os_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                files.sort()
                image_list = [f"{browser_base}/{img}" for img in files]
            except:
                pass

        if label == "Other___Tomato_disease":
            sub_folders = []
            try:
                # Find all disease folders inside 'Other___Tomato_disease'
                sub_folders = [f for f in os.listdir(full_os_path) if os.path.isdir(os.path.join(full_os_path, f))]
            except:
                pass
            
            library_items = []
            for f in sub_folders:
                # --- THE DEEP SCAN: GO INSIDE EACH SUB-FOLDER TO FIND IMAGES ---
                sub_os_path = os.path.join(full_os_path, f)
                sub_browser_base = f"/static/reference/Other___Tomato_disease/{f}"
                sub_images = []
                
                if os.path.exists(sub_os_path):
                    try:
                        # Look for 1.jpg, 2.jpg, 3.jpg inside THIS sub-folder
                        imgs = [img for img in os.listdir(sub_os_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        imgs.sort()
                        sub_images = [f"{sub_browser_base}/{img}" for img in imgs]
                    except:
                        pass
                
                # Build the library item WITH the image_list included!
                disease_info = raw_data.get(f, {})
                library_items.append({
                    "name": f.replace("_", " "),
                    "folder_path": sub_browser_base,
                    "image_list": sub_images[:3], # Adds the 3 sample images for the slider
                    # "treatment": raw_data.get(f, {})
                    "details": disease_info
                })

            return {
                "label": label,
                "display_name": "🔍 Other Tomato Diseases",
                "is_library": True,
                "library_data": library_items,
                "details": {}
            }
        else:
            # Standard return for Main Diseases (Early Blight, Late Blight, etc.)
            return {
                "label": label,
                "display_name": label.replace("Tomato___", "").replace("_", " "),
                "is_library": False,
                "image_list": image_list[:3],
                "details": raw_data
            }

    # --- STEP 5: FINAL RESPONSE TO LOVABLE (INCLUDES SAFETY) ---
    return {
        "status": result_status,
        "message": message,
        "user_image": f"{MY_PUBLIC_URL}/{image_path}", # The photo the user just took
        "history_data": history_record,
        
        "choice1": get_choice_data(label1),
        
        "choice2": get_choice_data(label2) if result_status == "ambiguous" else None,
        
        # This sends the Safety Mask/Gloves info to the frontend
        "safety_protocol": TREATMENT_DB.get("Human_Safety_Protocol", {})
    }