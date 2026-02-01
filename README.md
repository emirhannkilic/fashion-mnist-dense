md


# Fashion-MNIST Classification (Dense Neural Network)

This project implements a fully-connected neural network for multiclass image classification on the Fashion-MNIST dataset using TensorFlow and Keras.

The project follows a complete machine learning workflow including data preprocessing, training/validation/test splits, regularization, learning curve analysis, confusion matrix visualization, and error analysis.

---

## Dataset
- **Fashion-MNIST**
- 28×28 grayscale images
- 10 clothing categories:
  - T-shirt/top, Trouser, Pullover, Dress, Coat  
  - Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Project Structure

```text
.
├── main.py
├── src
│   ├── data.py        # dataset loading and train/val/test split
│   ├── model.py       # model architectures
│   ├── train.py       # training logic
│   ├── analysis.py    # evaluation and visualization
│   └── utils.py       # utilities and class names
├── models             # saved trained models
└── requirements.txt

---

## Model Architecture

### Baseline Model
- Flatten  
- Dense (128, ReLU)  
- Dense (10, Softmax)

### Regularized Model (Final)
- Flatten  
- Dense (256, ReLU, L2 regularization)  
- Dropout (0.3)  
- Dense (128, ReLU, L2 regularization)  
- Dropout (0.2)  
- Dense (10, Softmax)

---

## Training Setup
- Optimizer: Adam  
- Loss function: Sparse Categorical Crossentropy  
- Train / Validation / Test split:
  - Train: 55,000 samples
  - Validation: 5,000 samples (~8%)
  - Test: 10,000 samples
- Epochs: 10  
- Batch size: 128  

The validation set is evaluated during training to monitor overfitting and guide model selection.  
The test set is used only once for final evaluation.

---

## Evaluation
- Test Accuracy: ~86%
- Learning curves (training vs validation loss and accuracy)
- Confusion matrix
- Visualization of misclassified examples

---

## Error Analysis Summary
Most classification errors occur between visually similar classes:

- **Shirt ↔ T-shirt/top / Pullover / Coat**  
- **Sandal ↔ Sneaker**  
- **Pullover ↔ Coat**

These errors are expected for a fully-connected network, since spatial information is lost after the Flatten layer.  
A convolutional neural network (CNN) would likely improve performance by learning local spatial features.

---

## How to Run

1. Create and activate a virtual environment
2. Install dependencies:
	pip install -r requirements.txt 

3. Run training and evaluation:
	python main.py

