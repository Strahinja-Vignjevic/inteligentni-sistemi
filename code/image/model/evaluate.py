import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD FULL DATA
X = np.load("X.npy")
Y = np.load("Y.npy")

X = X[..., np.newaxis]

# SPLIT (ISTO KAO U TRAIN-U)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# LOAD MODEL
model = tf.keras.models.load_model("sax_model.h5")

# PREDICTION
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# METRICS
print(classification_report(Y_test, Y_pred_classes))

cm = confusion_matrix(Y_test, Y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["A", "A#", "B", "C", "C#", "C2", "D", "D#", "E", "F", "F#", "G", "G#"],
    yticklabels=["A", "A#", "B", "C", "C#", "C2", "D", "D#", "E", "F", "F#", "G", "G#"]
)
plt.title("Confusion Matrix — CNN")
plt.xlabel("Predvidjena nota")
plt.ylabel("Stvarna nota")
plt.tight_layout()
plt.savefig("confusion_matrix_cnn.png", dpi=150)
plt.show()