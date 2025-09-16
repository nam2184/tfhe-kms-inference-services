import h5py
from tensorflow.keras import backend as K
import os
import shutil
OUTPUTS_PATH = '.'


def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


def square(x):
    return x ** 2


def save_data_set(x, y, data_type, path=OUTPUTS_PATH, s=''):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Saving x_{} of shape {}".format(data_type, x.shape))
    xf = h5py.File(os.path.join(path, f'x_{data_type}{s}.h5'), 'w')
    xf.create_dataset('x_{}'.format(data_type), data=x)
    xf.close()

    yf = h5py.File(os.path.join(path, f'y_{data_type}{s}.h5'), 'w')
    yf.create_dataset('y_{}'.format(data_type), data=y)
    yf.close()


def save_weights(model, index=0, path=OUTPUTS_PATH):
    if not os.path.exists(path):
        os.mkdir(path)
    fname = os.path.join(path, "model_epoch_{:0>4}.h5".format(index))
    print("Saving weights to: " + fname)
    model.save_weights(fname)
    s = model.to_json()

    with open(os.path.join(path, f'model_epoch{index}.json'), 'w') as f:
        f.write(s)


def final_save_weights(model, index, path=OUTPUTS_PATH):
    weights_fname = os.path.join(path, "model_epoch_{:0>4}.h5".format(index))
    final_weights_fname = os.path.join(path, "model.h5")
    print("Saving weights to: " + final_weights_fname)
    shutil.copyfile(weights_fname, final_weights_fname)

    json_fname = os.path.join(path, f'model_epoch{index}.json')
    final_json_fname = os.path.join(path, "model.json")
    print("Saving model JSON to: " + final_json_fname)
    shutil.copyfile(json_fname, final_json_fname)


def square_activation(input):
    return input ** 2


def plot_history(model, score, lr, n_filters, dataset, batch_size, epochs, sgd_momentum=0, is_input_flattened=True):
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.plot(model.history.history['val_acc'], label='val_accuracy')
    plt.plot(model.history.history['acc'], label='accuracy')
    plt.legend()
    plt.grid()
    plt.ylabel("accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(model.history.history['val_loss'], label='val_loss')
    plt.plot(model.history.history['loss'], label='loss')
    plt.legend()
    plt.grid()
    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.savefig(os.path.join('figures', 'acc={:.3f}_loss={:.2f}_lr={}_filters={}_dataset={}_batch={}_epochs={}_momentum_{}_dense_{}_is_flattened={}.png'.
                format(score[1]*100, score[0], lr, n_filters, dataset, batch_size, epochs, sgd_momentum, dense_size, is_input_flattened)))
