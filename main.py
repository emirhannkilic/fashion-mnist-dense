from src.data import prepare_datasets
from src.model import create_regularized_model
from src.train import train_model
from src.analysis import (
    evaluate_model,
    save_model,
    plot_confusion_matrix,
    show_misclassified_examples,
    plot_loss_curves,
    plot_accuracy_curves,
)
from src.utils import CLASS_NAMES


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = prepare_datasets()

    model = create_regularized_model()

    print("\nModel summary:")
    model.summary()

    print("\nStarting training...")
    history = train_model(model, x_train, y_train, x_val, y_val)
    print("\nTraining finished.")

    evaluate_model(model, x_test, y_test)
    save_model(model)

    plot_confusion_matrix(model, x_test, y_test, CLASS_NAMES)
    show_misclassified_examples(model, x_test, y_test, CLASS_NAMES, n_samples=16)

    plot_loss_curves(history)
    plot_accuracy_curves(history)


if __name__ == "__main__":
    main()