import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD NEW TEST DATA
X_test = np.load("X.npy")
Y_test = np.load("Y.npy")

X_test = X_test[..., np.newaxis]

# LOAD MODEL
model = tf.keras.models.load_model("../../image/model/sax_model.h5")

# PREDICTION
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

# METRICS
print(classification_report(Y_test, Y_pred_classes))

cm = confusion_matrix(Y_test, Y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()