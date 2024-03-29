import tensorflow as tf


def get_model():
    """
    Builds the sequential NN model.
    """

    model_settings = {
        "type": "Convolutional with Batch Normalization",
        "residual_blocks": 10,
        "filters": 128,
    }

    model = tf.keras.Sequential()

    # Input
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu", input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    # Residual Blocks
    """
    # Conv and batch normalization layer
    for n in range(5):  # 10 blocks with 128 filters, 20 with 256 or 24 with 320
        model.add(tf.keras.layers.Conv2D(64, (1, 1), activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
    """

    """
    # Squeeze-Excitation Layer
    c = 1
    r = 4
    for n in range(5):
        model.add(tf.keras.GlobalAveragePooling2D())
        model.add(tf.keras.Dense(c/r, activation="relu"))
        model.add(tf.keras.Dense(c, activation="sigmoid"))
    """

    # Output
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model, model_settings

