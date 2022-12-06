import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
tf.random.set_seed(32)
np.random.seed(32)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import seaborn as sns

plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"



def scale(doppz, Max=9343, Min=36240):
    doppz_scaled = (doppz - Min) / (Max - Min)
    return doppz_scaled


def StackFrames(doppz, labels, frame_stack=10):
    max_index = doppz.shape[0] - frame_stack
    stacked_doppz = np.array([doppz[i:i + frame_stack] for i in range(max_index)]).transpose(0, 2, 3, 1)
    new_labels = np.array([labels[i + frame_stack - 1] for i in range(max_index)])
    return stacked_doppz, new_labels


class Dataset:
    def __init__(self, loc="micro_df.pkl", frame_stack=2):
        print(f"loading dataset from {loc}")
        df = pd.read_pickle(loc)
        df = df[df.Activity != '  '].reset_index()
        doppz = np.array(df['doppz'].values.tolist())
        label = df['Activity'].values
        dop_max, dop_min = doppz.max(), doppz.min()
        doppz_scaled_stacked, new_labels = StackFrames(scale(doppz, dop_max, dop_min), label, frame_stack)

        self.data, self.label = self.process(doppz_scaled_stacked, new_labels)

    def process(self, doppz_scaled_stacked, new_labels):
        return doppz_scaled_stacked, new_labels


def get_dataset():
    data = Dataset(loc='../micro_df.pkl',
                         frame_stack=2)
    
    lbl_map ={'laptop-typing': 0,
              'phone-talking': 1, 
              'phone-typing': 2,
              'sitting': 3
              }
    
    X_norm = data.data
    y = to_categorical(np.array(list(map(lambda e: lbl_map[e], data.label))), num_classes=4)
    
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def get_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(3,2),(2,1),padding="same",activation='relu',input_shape=(128,64,2)),
        tf.keras.layers.Conv2D(64,(3,3),(2,2),padding="same",activation='relu'),
        tf.keras.layers.Conv2D(96,(3,3),(2,2),padding="same",activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32,"relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(4,"softmax")
    ])
    return model


X_train, X_test, y_train, y_test=get_dataset()

model2 = get_model()

model2.load_weights('../micro_2fs.h5')
pred = model2.predict([X_test])

conf_matrix = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(pred, axis=1))
total = conf_matrix / conf_matrix.sum(axis=1).reshape(-1, 1)
total = np.round(total,2)
labels = ['laptop-typing', 'phone-talking', 'phone-typing', 'sitting']
df_cm = pd.DataFrame(total, index=[i for i in labels], columns=[i for i in labels])
sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues")
plt.savefig('micro_classification.pdf')
plt.show()
print(classification_report(np.argmax(y_test, axis=1), np.argmax(pred, axis=1)))
