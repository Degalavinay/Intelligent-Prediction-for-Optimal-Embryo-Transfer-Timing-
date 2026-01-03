# Endometrium Receptivity Analysis & Classification Project

This project focuses on analyzing and classifying endometrium receptivity using deep learning and morphological feature extraction. It consists of a data processing/analysis notebook and a Streamlit-based web application for inference and explainability.

## Project Structure

- **`App.py`**: A Streamlit application deployed for real-time inference. It uses a trained EfficientNet-B0 model to classify images as "Receptive" or "Non-Receptive".
    - **Features**: Single/Batch image upload, Model weight loading, JSON annotation support for cropping, Grad-CAM visualization for model explainability.
- **`P4_Code.ipynb`**: The main research and data pipeline notebook.
    - **Functionality**: dataset verification (COCO format), feature extraction (thickness, intensity, area), automated labeling based on percentile thresholds, and baseline classification using Decision Trees.
- **`Dataset_endometrium/`**: Directory containing the dataset images and COCO annotations (gitignored).

## Requirements

To run the code and application, you will need the following Python libraries:

```txt
streamlit
torch
torchvision
numpy
pandas
Pillow
matplotlib
seaborn
scikit-learn
pycocotools
scikit-image
scipy
```

You can install them using pip:

```bash
pip install streamlit torch torchvision numpy pandas Pillow matplotlib seaborn scikit-learn pycocotools scikit-image scipy
```

## Usage

### 1. Running the Web Application
To start the inference interface, run the following command in your terminal:

```bash
streamlit run App.py
```

This will open a local web server where you can:
- Upload your `.pth` model weights.
- Upload images for classification.
- View prediction probabilities and Grad-CAM heatmaps to understand model focus.

### 2. Running the Analysis Notebook
Open `P4_Code.ipynb` in Jupyter Notebook or Google Colab to explore the dataset statistics, regeneration of verification reports, and feature extraction logic.

## Features
- **EfficientNet-B0 Backbone**: Utilizes a powerful, efficient CNN architecture for image classification.
- **Explainable AI (XAI)**: Integrated Grad-CAM visualization allows users to see which parts of the endometrium image contributed to the decision.
- **Automated Data Processing**: Scripts to verify COCO datasets and extract quantitative morphological features automatically.
