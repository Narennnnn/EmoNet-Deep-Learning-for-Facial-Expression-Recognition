#  EmoNet-Deep-Learning-for-Facial-Expression-Recognition

This project implements a real-time facial expression detection system using a pre-trained deep learning model. The model can detect various facial expressions, such as happy, sad, angry, fear, disgust, surprise, and neutral, from a live video stream or an image.

## Dataset

The dataset used for training the facial expression detection model can be found at the following Google Drive link: [Facial Expression Detection Dataset](https://drive.google.com/drive/folders/10pcPEAN8res18o1BIyiHLP22XItYzRXJ?usp=sharing). It consists of labeled facial expression images to train the deep learning model.

## Installation 

To run this project, you need to follow these steps:

 Clone the repository:
git clone https://github.com/Narennnnn/EmoNet-Deep-Learning-for-Facial-Expression-Recognition.git

## Create and activate a virtual environment (optional but recommended):

python -m venv env

 **On Windows**
 
env\Scripts\activate

**On macOS and Linux**

source env/bin/activate

## Install the required Python libraries:
pip install -r requirements.txt

# Usage

To execute the real-time facial expression detection system, use the following command:

**Activate the virtual environment if not already activated**

**On Windows**

env\Scripts\activate

**On macOS and Linux**

source env/bin/activate

# Run the application using uvicorn

uvicorn main:app --host 0.0.0.0 --port 5000 --reload

The application will start, and you can access it through your web browser at http://localhost:5000/. The webcam will be activated to detect facial expressions in real-time. Detected emotions will be displayed along with a green rectangle around the detected face region.

 # Files and Directories
 
1. **FacialExpressionDetection.ipynb**: Jupyter notebook containing the steps to train and evaluate the facial expression detection model using deep learning.

2. **main.py**: Python script that defines the FastAPI application for real-time emotion detection.

3. **model.h5**: Pre-trained deep learning model for facial expression detection.

4. **templates**: Directory containing the HTML template (index.html) used for the web interface.

# Contributions
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.



