# Potato Leaf Disease Detection

## Overview
This project detects diseases in potato leaves using machine learning. It includes a model training pipeline and a Streamlit-based web application for real-time disease detection.

## Features
- Image preprocessing and augmentation
- Model training using a convolutional neural network (CNN)
- Streamlit-based web interface for easy image uploads and predictions
- Deployment-ready model

## Technologies Used
- Python
- TensorFlow/Keras
- Streamlit
- OpenCV
- NumPy & Pandas
- Matplotlib & Seaborn

## Dataset
The dataset consists of labeled images of potato leaves categorized into healthy and diseased classes. The images undergo preprocessing before training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SeaEscape/Potato_Leaf_Disease_Detection.git
   cd Potato-Leaf-Disease-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Training
Run the Jupyter Notebook to train the model:
```bash
jupyter notebook Train_potato_disease.ipynb
```
Ensure that your dataset and model saving paths are correctly configured.

## Running the Streamlit App
1. Update the model path in `web.py` to a relative or correct path.
2. Run the application:
   ```bash
   streamlit run web.py
   ```
3. Upload an image and get disease predictions.

## Fixes & Enhancements
- Updated model path for portability.
- Fixed duplicate `'Home'` condition in `web.py`.
- Added instructions for Streamlit deployment.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.
