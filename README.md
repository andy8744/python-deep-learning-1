# Rock Paper Scissors - Machine Learning Workshop

A beginner-friendly machine learning project for teaching image classification with deep learning. 
Modelled after our successful Deep Learning with Matlab Sesssion: https://github.com/andy8744/MATLAB_RockPaperScissors

Lab tutorial 1: https://www.kaggle.com/code/andy8744/dogs-vs-cats-beginner-cnn-with-augmentation
Lab tutorial 2: Follow the instructions below (laptop with webcam and python required!)


## 🎯 Project Overview

This project demonstrates:
- Image data collection via webcam
- Training CNN models from scratch
- Transfer learning with pre-trained models
- Real-time inference with webcam
- Robot control via serial communication

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Collect Training Data

Run the webcam capture tool to collect images:
```bash
python collect_images_via_webcam.py
```

- Click buttons to capture images for each gesture (Rock, Paper, Scissors, Null)
- Collect at least 50-100 images per class
- Images are saved to `data/` folder

### 2. Train a Model

**Option A: Simple CNN (from scratch)**
```bash
python train_model.py
```

**Option B: Transfer Learning (MobileNetV2)**
```bash
python train_model_transfer.py
```

Both will:
- Train on your collected images
- Save the best model
- Display training plots

### 3. Run Inference

**Standard webcam inference:**
```bash
python run_webcam.py
```

Or specify a model:
```bash
python run_webcam.py best_model_transfer.keras
```

**Robot hand control (requires Arduino):**
```bash
python run_webcam_robot.py
```

Update the serial port in the script before running.

## 📁 Project Structure

```
ml-with-python/
├── collect_images_via_webcam.py  # Data collection GUI
├── train_model.py                # Simple CNN training
├── train_model_transfer.py       # Transfer learning training
├── run_webcam.py                 # Standard inference
├── run_webcam_robot.py          # Robot control inference
├── plot_training.py              # Training visualization utilities
├── requirements.txt              # Python dependencies
└── data/                         # Training images (created on first run)
    ├── rock/
    ├── paper/
    ├── scissors/
    └── null/
```


## 📝 License

MIT License - Feel free to use for educational purposes!

## 🙏 Acknowledgments

Built for machine learning workshops and beginner education. Errm and thanks chatgpt :)

