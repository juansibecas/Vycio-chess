import numpy as np
import nn_model
from sklearn.model_selection import train_test_split
import load
from play import play
import os
import json

EPOCHS = 6
TEST_SIZE = 0.2


def run():
    data = load.load_data()

    model, settings = nn_model.get_model()

    labels = load.get_labels(data)

    inputs = load.get_inputs(data)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(inputs), np.array(labels), test_size=TEST_SIZE
    )

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    loss, acc = model.evaluate(x_test, y_test, verbose=2)

    # Save model on /models folder
    model_name = f'acc{acc:.4f}'
    model_dir = os.path.join('models', model_name)
    model.save(model_dir, save_format='tf')
    with open(f'{model_dir}.json', 'w') as fp:
        json.dump(settings, fp)

    # Play against it
    play(model)

    return data, model


if __name__ == '__main__':
    nn_data, nn_model = run()
