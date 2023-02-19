import numpy as np
import lstm
from sklearn.model_selection import train_test_split
from load import load_data
from play import play
import os
import json

EPOCHS = 6
TEST_SIZE = 0.2


def run():
    data = load_data()

    model, settings = lstm.get_model()

    labels = lstm.get_labels(data)

    inputs = lstm.get_inputs(data)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(inputs), np.array(labels), test_size=TEST_SIZE
    )

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    loss, acc = model.evaluate(x_test, y_test, verbose=2)

    model_name = f'acc{acc:.4f}'
    model_dir = os.path.join('models', model_name)
    model.save(model_dir, save_format='tf')
    with open(f'{model_dir}.json', 'w') as fp:
        json.dump(settings, fp)

    play(model)

    return data, model


if __name__ == '__main__':
    lstm_data, lstm_model = run()
