def train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=128):
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
    )
    return history