ORAL CANCER DETECTION USING CNN AND CLINICAL DATA

Overview
This project implements an automated oral cancer detection system using deep learning techniques. The system analyzes oral lesion images along with patient clinical data to predict the probability of oral cancer. A hybrid Convolutional Neural Network (CNN) model is used for improved accuracy, and Explainable AI techniques are applied to make the predictions interpretable.

Problem Statement
Early detection of oral cancer is critical for improving survival rates. Traditional diagnostic methods are time-consuming and depend heavily on expert examination. There is a need for an automated, accurate, and user-friendly system that can assist in early screening and decision support.

Proposed Solution
The proposed system combines image-based analysis using CNN with clinical features such as age, smoking status, and alcohol consumption. The extracted features are merged in a hybrid deep learning model to generate malignancy probability. Grad-CAM is used to visualize important regions in the lesion image that influence the prediction.

Features

Automated oral cancer detection

Hybrid CNN and clinical data model

Probability-based risk prediction

Explainable AI using Grad-CAM

User-friendly Streamlit web application

Technologies Used
Python
TensorFlow / Keras
OpenCV
NumPy
Joblib
Streamlit
Explainable AI (Grad-CAM)

Project Structure
Oral-Cancer-Detection
app.py – Streamlit application
preprocess.py – Image preprocessing functions
utils.py – Grad-CAM and visualization utilities
saved_models – Trained models and scaler
README.txt – Project documentation

How to Run

Install Python (version 3.10 or 3.11 recommended)

Install required dependencies

Open terminal in project directory

Run the application using the command:
streamlit run app.py

Input

Oral lesion image (JPG, JPEG, PNG)

Clinical data: age, smoking status, alcohol consumption

Output

Malignancy probability (%)

Risk classification (High, Low, Inconclusive)

Grad-CAM heatmap visualization

Applications

Early oral cancer screening

Medical decision support systems

Healthcare AI research

Academic projects

Future Scope

Integration of larger datasets

Multi-class oral disease classification

Mobile and cloud-based deployment

Conclusion
This project demonstrates how deep learning and clinical data can be combined to support early oral cancer detection. The hybrid approach improves accuracy, while Explainable AI enhances transparency and trust.

Disclaimer
This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

Author
Md Rasheed
B.Tech – Computer Science / AI / ML