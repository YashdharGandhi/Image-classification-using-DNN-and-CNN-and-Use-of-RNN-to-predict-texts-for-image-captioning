# Image Classification using DNN and CNN with RNN-based Image Captioning

## Project Overview
This project implements an end-to-end deep learning pipeline that addresses two main tasks:
- **Image Classification:** Utilizes Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN) to classify images.
- **Image Captioning:** Leverages a Recurrent Neural Network (RNN) to generate descriptive captions for images.

The project demonstrates how to combine computer vision and natural language processing techniques using Python and PyTorch.

## Data Description
- **Dataset:** The project uses a compressed zip file named **COCO Dataset.zip** located in the `data` directory of the repository. This dataset contains a collection of images along with their annotations.
- **Preprocessing:** Images are resized, normalized, and augmented to meet the training requirements of the deep learning models.
- **Annotations and Code:** The repository includes a notebook in the `Notebook` directory that contains code snippets for model implementation, training, and evaluation.

## Tasks
- **Model Implementation:**
  - Design and implement a DNN and a CNN for image classification.
  - Develop an RNN-based model for image captioning.
- **Training and Evaluation:**
  - Split the dataset into training and validation sets.
  - Train the models while monitoring loss and accuracy.
  - Plot training and validation performance metrics.
- **Overfitting Mitigation:**
  - Apply data augmentation and dropout techniques to reduce overfitting.
- **Repository Management:**
  - Organize the project into a clear directory structure and manage it using Git.

## Key Findings
- **Image Classification:**  
  - The CNN model demonstrated progressive improvement in both training and validation accuracy.
  - Techniques such as dropout and data augmentation effectively reduced overfitting.
- **Image Captioning:**  
  - The RNN-based captioning module was successful in generating descriptive captions from image features.
- **Overall Impact:**  
  - The integration of multiple deep learning techniques offers a robust framework for addressing complex tasks in both image classification and captioning, with potential for further enhancement through hyperparameter tuning and model refinements.

## Requirements to Run
- **Programming Language:** Python 3.6 or higher
- **Libraries:**  
  - PyTorch (and torchvision)
  - NumPy
  - Matplotlib
  - OpenCV (optional, for image processing)
  - PIL (Python Imaging Library)
- **Hardware:**  
  - A GPU is recommended for faster training (CPU-only execution is supported).
- **Environment:**  
  - Git Bash for repository management.
  - Jupyter Notebook or JupyterLab for running the provided notebook.
##Author
**Yashdhar Gandhi**
##License
This project is licensed under the MIT License.
