from tensorflow.keras import layers, models, regularizers


def create_basic_model():
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_regularized_model(l2_lambda=0.001, d1=0.3, d2=0.2):
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_lambda),
            ),
            layers.Dropout(d1),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_lambda),
            ),
            layers.Dropout(d2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model