# Brain Tumor Classification

A lightweight machine learning application to classify brain MRI images into three tumour types using deep learning.

## Models Evaluated
- **Custom CNN**: Compact 3-layer convolutional network
- **ResNet18**: Transfer learning with ResNet18 backbone
- **EfficientNet-B0**: Pretrained EfficientNet-B0 fine-tuned for MRI classification

## Tumour Classes
- **Class 1 (1.0)**: Meningioma
- **Class 2 (2.0)**: Glioma
- **Class 3 (3.0)**: Pituitary

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`, upload an MRI image (`.jpg`, `.jpeg`, `.png`), and compare the predictions from all three models.
