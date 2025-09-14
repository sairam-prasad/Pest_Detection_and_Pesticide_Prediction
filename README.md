# Pest_Detection_and_Pesticide_Prediction

The model proposes a novel approach to automatic pest detection and pesticide recommendation using a CNN, which is built on MobileNetV2 architecture. It will integrate an interactive web-based UI implemented with React and a backend service designed with Flask. Users can upload images of pests through the UI, which the backend will process and analyze for pest classification and recommend suitable pesticides along with alternative solutions. The model helps farmers and agricultural professionals with an easy, accurate, and scalable pest detection system.

#System Architecture
The architecture of the system is divided into three major parts: frontend, backend, and deep learning model.

#Frontend:

Implemented using: ReactJS
Functionality: This provides a very user-friendly interface through which users can upload images of pests and display the results.
User Experience: Once an image is uploaded, it displays the classification of the pest along with recommended pesticides and alternative solutions.

Backend:

Implemented using: Flask

Responsibilities:
Uploads images and temporarily stores them in a secure directory. Preprocesses the images to be compatible with the deep learning model. Loads the trained MobileNetV2 model and predicts the type of pest. Retrieves pesticide recommendations and alternative solutions from a structured pesticide dataset. Integration: The backend handles all communications between the frontend and the model, ensuring a seamless experience. 

Model:

Architecture: MobileNetV2 is a pre-trained convolutional neural network optimized for mobile and embedded vision applications.
Training Dataset: The dataset consists of “write about dataset”.
Preprocessing Pipeline:
Images are resized to
224×224 pixels.
Pixel values are normalized to the range [0, 1].

Layers:
Base Model: MobileNetV2, including ImageNet pre-trained weights.
Custom Layers:
A Global Average Pooling layer to reduce the dimensionality.
A Dense Layer with 128 neurons and L2-regularization (
𝜆 = 0.05 to avoid overfitting.

A Dropout Layer with a rate of 0.2 to avoid overfitting.
An output Dense Layer with a softmax activation function for multi-class pest classification.
Optimizer: Adam optimizer with a learning rate of 10−5
and an exponential decay schedule.
Loss Function: Categorical Crossentropy with label smoothing ( 𝛼 =0.1)

Callbacks:
Early Stopping: Monitors validation loss to prevent overfitting.
Model Checkpoint: Saves the best model during training.
Learning Rate Scheduler: Reduces learning rate by 50% if validation loss plateaus.
Training and Validation
A robust pipeline with rotation, zoom, width/height shift, brightness adjustment, and horizontal flipping enhances the dataset for improved model generalization.
Then, the dataset is split into training and validation subsets:
Training: 70%
Validation: 30%
Model training for 50 epochs, batch size of 32.
Results:
Training Accuracy:


>90%
Validation Accuracy:
>
>85%
Validation Loss: Minimal to ensure low overfitting rate.
Workflow

Image Upload:
User will upload images of pests through the React-based frontend.
The image is sent to the Flask backend for processing. Image Preprocessing: Images are resized to 224×224 pixels and normalized. Model Inference: It passes the preprocessed image through the MobileNetV2 model for pest classification. Pesticide Recommendation: With the predicted class of a pest, the system will query a structured dataset for fetching pesticide and alternative solution recommendations. Results Delivery: The final results are returned to the client-side by mentioning the name of the pest, recommending pesticide usage, and finding alternative remedies. Key Features
User Interactive Interface: Enhancing users' experience, it can upload images with ease to interpret the results.

Scalable: lightweight architecture of MobileNetV2 guarantees that a system can work on edge devices and high-performance servers.
Custom Dataset: Pesticide Dataset is being used to offer actionable results.
Robust Preprocessing: Ensures the quality of images and removes corrupted data during training and inference.

