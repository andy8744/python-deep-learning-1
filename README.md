# Rock Paper Scissors - Machine Learning Workshop

A beginner-friendly machine learning project for teaching image classification with deep learning. Includes webcam data collection, model training, and real-time inference with optional robot hand control.

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

## 🎓 For Workshop Instructors

This project is designed for beginners with minimal ML experience:

1. **Simple workflow**: Collect → Train → Test
2. **Visual feedback**: All tools have GUI/visual output
3. **Two approaches**: Compare simple CNN vs transfer learning
4. **Hands-on**: Physical robot demo (optional)
5. **Clean code**: Minimal complexity, well-commented

## 🤖 Robot Hand Setup (Optional)

If using the Arduino robot hand:

1. Upload the Arduino sketch to your board
2. Update `SERIAL_PORT` in `run_webcam_robot.py`
3. Run the robot inference script

Commands sent:
- `r` = rock
- `p` = paper
- `s` = scissors
- `d` = default/null position

## 🔧 Configuration

Key parameters you can adjust:

**Data Collection:**
- Image size: 224x224 (in collect script)
- Categories: rock, paper, scissors, null

**Training:**
- Batch size: 8 (adjust based on memory)
- Epochs: 20-50
- Learning rate: 0.001
- Train/validation split: 70/30

**Inference:**
- Confidence threshold: 45% (robot control)
- Command cooldown: 30 frames (~1 second)

## 📊 Expected Results

- Simple CNN: ~80-90% accuracy with 100+ images per class
- Transfer learning: ~90-95% accuracy with 50+ images per class

## 🐛 Troubleshooting

**Model won't load:**
- Check that the model file exists
- Try removing `safe_mode=False` if using newer TensorFlow

**Low accuracy:**
- Collect more diverse training images
- Try transfer learning approach
- Increase training epochs

**Arduino not connecting:**
- Check serial port path
- Verify baud rate (9600)
- Try unplugging/replugging USB

## 📝 License

MIT License - Feel free to use for educational purposes!

## 🙏 Acknowledgments

Built for machine learning workshops and beginner education.

