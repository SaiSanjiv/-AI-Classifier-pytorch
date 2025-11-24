---
title: Animal Classifier 🐾
emoji: 🐾
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
python_version: 3.9
---

# Animal Classifier 🐾

This project is a deep learning app that classifies images as Cat, Dog, or Panda using a fine-tuned ResNet50 model. It includes a clean Streamlit interface and is fully optimized for Hugging Face Spaces.

## Features

### Smart Image Classification
- Upload JPG, JPEG, PNG, or WebP images
- Real-time predictions with confidence scores
- Plotly charts to show probabilities
- Automatic model download from Hugging Face
- Works with different image sizes and formats

### Analytics
- Detailed precision, recall, and F1-scores for each class
- Confusion matrix with interactive heatmap
- Training history plots for loss and accuracy
- Model architecture breakdown
- Export metrics as CSV or JSON

### Interface
- Clean, modern UI with custom styling
- Works well on desktop and mobile
- Confidence gauge charts
- Sidebar navigation
- Live model status display

## Model Architecture

This project uses transfer learning with ResNet50:

- Base model: ResNet50 (ImageNet pre-trained)
- Fine-tuned layers: Layer4 + custom classifier
- Classifier:
  - Linear 2048 → 512 + ReLU + Dropout 0.7
  - Linear 512 → 128 + ReLU + Dropout 0.3
  - Linear 128 → 3 (output layer)
- Classes: Cat, Dog, Panda
- Model auto-downloads on first run
- Runs on CPU or GPU

## Performance

- Test accuracy: 99.33%
- Cat F1-score: 99.00%
- Dog F1-score: 98.99%
- Panda F1-score: 100.00%
- Macro average F1-score: 99.33%
- Weighted average F1-score: 99.33%

## Project Structure

Animal-classification-deploy/
├── Dockerfile
├── app.py
├── cat-dog-pandas-classification.ipynb
├── model.pth
├── metrics.json
├── requirements.txt
├── confusion_matrix.png
└── datasplit.py


## Quick Start

### Local Setup

1. Clone the repository:
git clone <your-repo>
cd Animal-classification-deploy


2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app.py

4. Open: http://localhost:8501

### Docker Deployment

1. Build the Docker image:
docker build -t animal-classifier .

2. Run the container:
docker run -p 7860:7860 animal-classifier

3. Open: http://localhost:7860

## Hugging Face Spaces Deployment (Recommended)

### Step 1: Create a Space
- Go to Hugging Face Spaces
- Select Docker as SDK
- Set app_port to 7860

### Step 2: Upload Files
git clone https://huggingface.co/spaces/yourusername/animal-classifier
cd animal-classifier

cp Animal-classification-deploy/* .
git lfs install
git lfs track "*.pth"
git add .
git commit -m "Deploy animal classifier"
git push


### Step 3: Configuration
- Dockerfile optimized for CPU
- requirements.txt includes lightweight packages
- README contains the correct Space metadata

## Technical Details

### Dependencies
streamlit>=1.34.0
torch==2.0.0+cpu
torchvision==0.15.1+cpu
numpy==1.24.3
pandas==2.0.3
pillow==10.0.0
plotly==5.15.0
matplotlib==3.9.2

### Training Settings
- Optimizer: AdamW (lr=5e-6)
- Loss: CrossEntropy with label smoothing
- Batch size: 32
- Epochs: 20 (with early stopping)
- Augmentation: crop, flip, rotation, color jitter

### Model Specs
- Parameters: ~24M total
- Trainable: ~16M
- Input: 224x224 RGB
- Normalized using ImageNet stats

## Usage

### Basic Classification
1. Open the app
2. Go to the Prediction tab
3. Upload an image
4. Press "Classify Image"
5. View confidence scores and charts

### Performance Analysis
- Open the Model Metrics tab
- View performance summaries
- Check confusion matrix
- Review training curves

### Programmatic Model Loading
import torch
from torchvision import models

model = models.resnet50(weights=None)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()


## Dataset

- Cat, Dog, Panda dataset (balanced)
- 600 test images (200 per class)
- Images resized to 224x224 during preprocessing
- Train/Val/Test splits included in the notebook

## Development

### Local Development
git clone <your-repo-url>
cd Animal-classification-deploy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py


### Retraining the Model
Open the notebook:
jupyter notebook cat-dog-pandas-classification.ipynb


### Adding New Classes
- Update dataset folders
- Update class names in app.py and metrics.json
- Retrain the model

## Deployment Options

### Hugging Face Spaces
- Free hosting
- HTTPS included
- Easy sharing

### Docker Self-hosting
docker build -t animal-classifier .
docker run -d -p 7860:7860 animal-classifier

markdown
Copy code

### Cloud Platforms
- AWS ECS / Fargate
- Google Cloud Run
- Azure Container Instances
- Railway

## Highlights
- 99.33% accuracy
- Dockerized and production-ready
- Clean UI with interactive visualizations
- Full analytics included
- Model auto-downloads
- Works on mobile
- Fast inference even on CPU

## Contributing
1. Fork the repo  
2. Create branch: `git checkout -b feature-name`  
3. Commit: `git commit -am "Add feature"`  
4. Push: `git push origin feature-name`  
5. Open a pull request  

## License
Apache 2.0 License

---

Built using PyTorch, Streamlit, and Docker by Sai Sanjiv R.
