import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .utils import ensure_dir


def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    return test_loss, test_acc


def save_model(model, filename="models/regularized_fashion_mnist.keras"):
    ensure_dir("models")
    model.save(filename)
    print(f"\nModel saved to {filename}")


def plot_loss_curves(history):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_curves(history):
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(train_acc) + 1)

    plt.figure()
    plt.plot(epochs, train_acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(model, x_test, y_test, class_names):
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def show_misclassified_examples(model, x_test, y_test, class_names, n_samples=16):
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    misclassified_idx = np.where(y_pred != y_test)[0]
    if len(misclassified_idx) == 0:
        print("No misclassified examples found.")
        return

    misclassified_idx = misclassified_idx[:n_samples]

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(misclassified_idx):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_test[idx], cmap="gray")
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()