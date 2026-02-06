# 🏺 Ready-to-Use Archaeological Datasets

Here are free public datasets you can use to train your model immediately.

## 🏆 Top Recommendation: Roboflow Universe
**Roboflow Universe** is the best place to find ready-to-use YOLOv8 datasets.

1. **Go to:** [universe.roboflow.com](https://universe.roboflow.com/search?q=archaeology)
2. **Search for:**
   - "Archaeology"
   - "Pottery shards"
   - "Coins"
   - "Stone tools"

### 📥 How to Download from Roboflow:
1. Click "Download Dataset"
2. Select Format: **YOLOv8**
3. Select "Download Zip"
4. Extract the zip file to your `project1/dataset/` folder

---

## 📚 Specific Datasets found

### 1. Stone Tools & Rock Art
- **Stone Tool Dataset** (Roboflow)
  - Contains flaked stone artifacts
  - Good for class: `Stone`
  - [Search Link](https://universe.roboflow.com/search?q=stone%20tools)

### 2. Pottery & Ceramics
- **Archaeological Pottery**
  - Contains sherds and vessels
  - Good for class: `Pottery` (or mapped to `Stone`/`Plastic` if relevant)
  - [Search Link](https://universe.roboflow.com/search?q=pottery)

### 3. Coins (Gold/Metal)
- **Roman Coins Dataset**
  - Contains gold/silver/bronze coins
  - Good for class: `Gold` / `Metal`
  - [Search Link](https://universe.roboflow.com/search?q=coins)

### 4. General Excavation
- **Archaeology Sites**
  - Aerial views and excavation pits
  - Good for class: `Soil` / Context
  
---

## 🌍 Other Sources

### Kaggle Datasets
- **Creation of Artifacts**
- **High-Fidelity Cultural Heritage**
- [Search Kaggle Archaeology](https://www.kaggle.com/search?q=archaeology+yolo)

### Academic Repositories
- **Zenodo** (Often hosts scientific datasets for stone tools)
- **Figshare** (Search for "Lithics" or "Ceramics")

---

## 🏛️ Archaeological Databases (For Raw Images)
*Note: These sources provide excellent images but usually require **manual labeling** (drawing boxes).*

### 1. Arachne Object Database (Recommended)
- **Best for:** Pottery, Stone objects, Metal artifacts.
- **How to use:**
  1. Search for an object type (e.g., "terra sigillata").
  2. Download the high-quality images.
  3. Upload to **train_manager.py** and label them.

### 2. Archaeology Data Service (ADS) & AADA
- **Best for:** UK/European finds, site context images.
- **Usage:** Good for finding specific regional artifacts (e.g., Finnish stone tools), but requires searching through reports to find image plates.

### ❌ Less Suitable for this App
- **DATAMESOP / Wanyika / MURR:** These are primarily **text/chemical data** (Excel/CSV). They don't contain enough images for training a vision AI.

---

## 🚀 How to use these datasets with this App?

1. **Download** the dataset zip (YOLOv8 format)
2. **Extract** contents
3. **Move** images and labels to:
   - `project1/dataset/images`
   - `project1/dataset/labels`
4. **Run Training** using `train_manager.py` or command line

**Tip:** You can combine multiple datasets! Just put all images in the same folder and update your `data.yaml` class list.
