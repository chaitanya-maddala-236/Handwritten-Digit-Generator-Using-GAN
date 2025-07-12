# Handwritten Digit Generator Using GAN

A web application that generates handwritten-style digits (0-9) using a Generative Adversarial Network (GAN) trained on the MNIST dataset.

## 🚀 Live Demo

**[Try the Live App Here](https://japanmetiai.streamlit.app/)**

## ✨ Features

- **Interactive Digit Selection**: Choose any digit from 0-9 with intuitive button interface
- **Batch Generation**: Creates 5 unique variations per digit for comparison
- **Real-time Display**: View generated images instantly with no waiting
- **MNIST-style Output**: Produces 28×28 grayscale images similar to the MNIST dataset
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **High-Quality Generation**: Creates diverse, recognizable handwritten digits

## 🧠 Model Architecture

### GAN Type
- **Model**: Conditional GAN (cGAN)
- **Framework**: PyTorch
- **Dataset**: MNIST (60,000 training images)
- **Training**: 50 epochs on Google Colab T4 GPU
- **Input**: 100D noise vector + digit label
- **Output**: 28×28 grayscale images

### Generator Architecture
```
Input: [100D noise + 10D label embedding] → 200D
↓
Linear(200, 256) + LeakyReLU + BatchNorm
↓
Linear(256, 512) + LeakyReLU + BatchNorm
↓
Linear(512, 1024) + LeakyReLU + BatchNorm
↓
Linear(1024, 784) + Tanh
↓
Reshape → 28×28 image
```

### Discriminator Architecture
```
Input: [784D image + 784D label embedding] → 1568D
↓
Linear(1568, 1024) + LeakyReLU + Dropout(0.3)
↓
Linear(1024, 512) + LeakyReLU + Dropout(0.3)
↓
Linear(512, 256) + LeakyReLU + Dropout(0.3)
↓
Linear(256, 1) + Sigmoid
```

## 📁 Project Structure

```
digit-generator/
├── mnist_gan_training.py    # GAN training script (for Google Colab)
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── generator.pth           # Trained model weights (after training)
└── README.md               # This file
```

## 🔧 Installation & Setup

### Option 1: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/chaitanya-maddala-236/Handwritten-Digit-Generator-Using-GAN.git
   cd Handwritten-Digit-Generator-Using-GAN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

### Option 2: Train Your Own Model

1. **Open Google Colab and create a new notebook**
2. **Upload `mnist_gan_training.py` to Colab**
3. **Run the training script**:
   ```python
   !python mnist_gan_training.py
   ```
4. **Download the generated `generator.pth` file**
5. **Training time**: ~2-3 hours on T4 GPU

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Public Access**: Your app will be available at a public URL

## 🎯 How to Use

1. **Visit the web app** (local or deployed URL)
2. **Select a digit** (0-9) by clicking the digit buttons
3. **Click "Generate 5 Images"** to create variations
4. **View results** - both enlarged and original 28×28 versions
5. **Try different digits** to see various generation results

## ⚡ Performance

- **Generation Time**: 1-3 seconds per batch of 5 images
- **Model Size**: ~2.5MB (generator.pth)
- **Memory Usage**: ~100MB RAM
- **Cold Start**: 10-30 seconds (first load)
- **Concurrent Users**: Supports multiple users simultaneously

## 💻 System Requirements

### For Training
- **Platform**: Google Colab (free tier)
- **GPU**: T4 GPU (recommended)
- **Memory**: 12GB RAM
- **Storage**: 2GB for dataset and models

### For Deployment
- **Platform**: Streamlit Cloud / Heroku / Railway
- **Python**: 3.8+
- **Memory**: 512MB+ RAM
- **Dependencies**: See `requirements.txt`

## 🎨 Generation Examples

The GAN generates diverse, recognizable handwritten digits:
- **Digit 0**: Oval shapes with variations in thickness and angle
- **Digit 1**: Vertical lines with different slopes and styles
- **Digit 2**: Curved and straight segments forming "2" shapes
- **Digit 3**: Horizontal segments with right-side curves
- **...and so on for digits 4-9**

Each generation produces 5 unique variations that are recognizable and diverse.

## 🔍 Troubleshooting

### Common Issues

1. **Missing Model File**
   - Ensure `generator.pth` is in the same directory as `app.py`
   - Check file size (should be ~2-5MB)
   - If missing, app will use mock generation

2. **Deployment Issues**
   - Verify all files are committed to git
   - Check repository is public for Streamlit Cloud
   - Ensure `requirements.txt` has correct versions

3. **Memory Issues**
   - Use CPU version of PyTorch: `torch==2.0.0+cpu`
   - Reduce batch size in training if needed

## 📄 License

This project is for educational purposes. The MNIST dataset is publicly available and commonly used for machine learning research.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **PyTorch**: Facebook AI Research
- **Streamlit**: Streamlit Inc.
- **Training Platform**: Google Colab

## 👤 Author

**Chaitanya Maddala**
- GitHub: [@chaitanya-maddala-236](https://github.com/chaitanya-maddala-236)
- Project Link: [https://github.com/chaitanya-maddala-236/Handwritten-Digit-Generator-Using-GAN](https://github.com/chaitanya-maddala-236/Handwritten-Digit-Generator-Using-GAN)

---

⭐ If you found this project helpful, please give it a star on GitHub!
