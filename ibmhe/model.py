import os
import tensorflow as tf
from tensorflow.keras import layers, models
import dataset 

def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 classes
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, batch_size, epochs, x_val, y_val):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

if __name__ == "__main__":
    model_8x8 = build_model((8, 8, 1))
    model_80x80 = build_model((80, 80, 1))
    model_160x160 = build_model((160, 160, 1))

    # Print summaries
    print("8x8 Model Summary:")
    model_8x8.summary()

    print("\n80x80 Model Summary:")
    model_80x80.summary()

    print("\n160x160 Model Summary:")
    model_160x160.summary()
    project_root = os.path.dirname(os.getcwd())
    x_train, y_train, x_val, y_val = dataset.split_and_preprocess(project_root + "/dataset", size=(80,80))
    train_model(model_80x80, x_train, y_train, batch_size = 64, epochs = 30, x_val = x_val, y_val = y_val)

    model_json = model_80x80.to_json()
    with open(project_root + "/models/model80x80_architecture.json", "w") as json_file:
        json_file.write(model_json)

    # Save weights as HDF5
    model_80x80.save_weights(project_root+ "/models/model80x80_weights.h5")
    print("Model architecture and weights saved!")
