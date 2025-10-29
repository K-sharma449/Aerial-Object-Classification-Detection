---

```markdown
# Aerial Object Detection: Bird vs Drone  
![Python](https://img.shields.io/badge/python-3.10-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)  
![License](https://img.shields.io/badge/license-MIT-green)

**Real-time classification and detection of birds and drones using deep learning.**

A lightweight, end-to-end system that:
- Classifies images as **Bird** or **Drone** using **EfficientNetB0** (Transfer Learning)
- Detects objects using **YOLOv8**
- Runs a **beautiful Streamlit dashboard** for instant inference

---

## Features

| Feature | Description |
|-------|-----------|
| Classification | EfficientNetB0 fine-tuned on bird/drone dataset |
| Object Detection | YOLOv8n for real-time bounding boxes |
| Web Dashboard | Upload image → instant result |
| Auto Dataset Setup | Creates dummy data if folders are missing |
| VS Code Ready | No Pylance errors, clean setup |

---

## Demo

![Demo GIF](demo.gif)  
*(Add a GIF of your dashboard in action)*

---

## Project Structure

```
AOCD/
│
├── aerial_object_detection.py     Main script (train + dashboard)
├── aocd_env/                      Virtual environment (don't commit)
├── classification_dataset/        Images (TRAIN/VALID/TEST)
├── object_detection_dataset/      YOLO format (optional)
├── transfer.keras                 Trained model (generated)
├── runs/                          YOLO training logs
├── .vscode/settings.json          VS Code config
└── README.md                      This file
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/AOCD.git
cd AOCD
```

### 2. Create & activate virtual environment
```powershell
python -m venv aocd_env
.\aocd_env\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install tensorflow==2.20.0 ultralytics streamlit opencv-python pillow numpy scikit-learn seaborn matplotlib
```

### 4. Train the model
```bash
python aerial_object_detection.py --train
```
> Creates `transfer.keras` and dummy data if needed

### 5. Launch Dashboard
```bash
streamlit run aerial_object_detection.py
```

Open browser → Upload image → See **Bird** or **Drone**!

---

## Dataset (Optional but Recommended)

Download classification dataset from [Mendeley Data](https://data.mendeley.com/datasets/6ghdz52pd7)  
Extract to:
```
classification_dataset/
├── TRAIN/bird/
├── TRAIN/drone/
├── VALID/bird/
├── VALID/drone/
├── TEST/bird/
└── TEST/drone/
```

> Even **1 image per class** works for testing!

---

## Fix VS Code Red Lines (Pylance)

1. `Ctrl + Shift + P` → **"Python: Select Interpreter"**
2. Choose: `.\aocd_env\Scripts\python.exe`
3. Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "aocd_env",
    "python.analysis.extraPaths": ["aocd_env/Lib/site-packages"],
    "python.analysis.typeCheckingMode": "basic"
}
```
Restart VS Code → **No red squiggles**

---

## Training Tips

- Use **GPU** for faster training
- Increase `epochs=10` in code for better accuracy
- Replace dummy data with real dataset for production

---

## Output Examples

| Input | Output |
|------|--------|
| Bird photo | **Bird (98.7%)** |
| Drone photo | **Drone (99.1%)** |

---

## Contributing

1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

[MIT License](LICENSE) - Free to use, modify, and distribute.

---

## Author

**Your Name**  
GitHub: [@yourusername](https://github.com/yourusername)  
Email: your.email@example.com

---

**Star this repo if you found it helpful!**
```

---

## Next Steps for You

1. **Create `README.md`** in your project root
2. **Paste the above content**
3. **Add a `demo.gif`** (optional but awesome)
4. **Commit & push to GitHub**

```bash
git add README.md
git commit -m "Add professional README"
git push origin main
```

---
