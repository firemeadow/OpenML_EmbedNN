from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow import keras
import matplotlib.pyplot as plt

# Lucas Invernizzi
# Embedded Neural Network Classifier for any OpenML dataset
# NN layer sizes need to be adapted for each dataset
# Loss function needs to be adapted for non-binary classification
# Labels need to mapped to binary for each dataset

# Creates dataset, batches
def df_to_dataset(df, batch_size):
    df = df.copy()
    labels = df.pop(df.columns[df.shape[1] - 1])
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.batch(batch_size)
    return ds


# Loads, shuffles, batches, creates embedding layer for the given dataset ID and batch size
# Returns training, validation, and testing sets
# Throws SettingWithCopyWarning, but I am using the functionality correctly so just ignore
def load_data(dataset, batch_size, num_unique_cat, tt_split, tv_split, num_embed_dims):
    print("Loading Data...")
    out = fetch_openml(data_id=dataset, as_frame=True)
    data = out['data']
    labels = out['target']
    headers = data.columns

    # Dict which holds every categorical attribute's possible values
    cat_maps = {}
    # For every attribute
    for i, col in data.items():
        # Fill NaN values
        if np.sum(col.isnull()) > 0:
            # If categorical, fill with new category
            if col.dtype.name == 'category':
                data[i] = data[i].cat.add_categories('Missing')
                data[i].fillna('Missing')
            # Otherwise, fill with column median
            else:
                data[i].fillna(np.median(col))

        # Represent categorical attributes as strings and store their possible values
        if col.dtype.name == 'category' or col.unique().shape[0] <= num_unique_cat:
            data[i] = col.astype(str)
            cat_maps[i] = data[i].unique()

    # Uncomment the two lines below, change label_map to match possible label values to binary if not already done
    # label_map = {-1 : 0, 1 : 1}
    labels = labels.astype(int)
    # labels = labels.map(label_map)

    # Combine data, labels, shuffle
    data = pd.concat([data, labels], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)

    # Split train, validation, test sets, then wrap in tensorflow dataset object and batch
    train, test = train_test_split(data, test_size=tt_split)
    train, valid = train_test_split(train, test_size=tv_split)
    train_ds = df_to_dataset(train, batch_size)
    val_ds = df_to_dataset(valid, batch_size)
    test_ds = df_to_dataset(test, batch_size)

    # Create Embedding Layer
    feature_columns = []
    # For every attribute
    for header in headers:
        # If the attribute is categorical, create an embedding column
        if header in cat_maps.keys():
            # Maps column to integer
            cat = fc.categorical_column_with_vocabulary_list(header, cat_maps[header])
            # Embeds integers to the specified number of dimensions
            embed = fc.embedding_column(cat, dimension=num_embed_dims)
            feature_columns.append(embed)
        else:
            feature_columns.append(fc.numeric_column(header))

    # Combine all feature columns into the first layer of the nn
    feature_layer = keras.layers.DenseFeatures(feature_columns)
    print('Data Loading Complete.')
    return train_ds, val_ds, test_ds, feature_layer


# Makes a linear model
def make_model(feature_layer, lr):
    model = keras.Sequential([
        feature_layer,
        keras.layers.Dense(100, activation='sigmoid'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(50, activation='sigmoid'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(25, activation='sigmoid'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='sigmoid'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(5, activation='sigmoid'),
        keras.layers.Dense(1)
    ])

    opt = keras.optimizers.Adam(lr)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)

    model.compile(opt, loss, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 30
    num_epochs = 10
    lr = 0.003  # Learning Rate
    dataset_id = 981  # ID of OpenML dataset to use
    tt_split = 0.2  # Ratio to split training and testing
    tv_split = 0.2  # Ratio to split training and validation
    num_unique_cat = 20  # Max number of unique values for an attribute to be considered categorical
    num_embed_dims = 16  # Number of dimensions the embedding layer embeds

    # Creates training, validation, test sets along with the first embedding layer for the dataset id
    train_data, val_data, test_data, feature_layer = load_data(dataset_id,
                                                               batch_size,
                                                               num_unique_cat,
                                                               tt_split,
                                                               tv_split,
                                                               num_embed_dims)

    model = make_model(feature_layer, lr)

    # Training loop
    hist = model.fit(train_data, epochs=num_epochs, verbose=2, validation_data=val_data).history

    # Test Model
    test_loss, test_acc = model.evaluate(test_data, verbose=2)

    print("Test Loss: " + str(np.round(test_loss * 100, 2)) + "%.")
    print("Test Accuracy: " + str(np.round(test_acc * 100, 2)) + "%.")

    # Plots training and validation losses and accuracies over all epochs
    # Also plots test loss and accuracy as horizontal lines for reference
    fig, axs = plt.subplots()
    axs.plot(hist['loss'], label='loss', color='r')
    axs.plot(hist['accuracy'], label='accuracy', color='b')
    axs.plot(hist['val_loss'], label='val_loss', color='y')
    axs.plot(hist['val_accuracy'], label='val_accuracy', color='g')
    axs.axhline(y=test_loss, label='test_loss', color='m')
    axs.axhline(y=test_acc, label='test_acc', color='c')
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    axs.legend(loc='best')
    plt.show()
