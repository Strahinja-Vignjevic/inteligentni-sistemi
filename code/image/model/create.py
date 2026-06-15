import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================

X = np.load("X.npy")
Y = np.load("Y.npy")

X = X[..., np.newaxis]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# =========================
# 2. TRAIN / VAL / TEST SPLIT
# =========================

# 70% train, 30% temp
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

# 15% val, 15% test
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp,
    test_size=0.5,
    random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# =========================
# 3. MODEL
# =========================

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(len(np.unique(Y)), activation='softmax')
])

# =========================
# 4. COMPILE
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. TRAINING (WITH VALIDATION)
# =========================

history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, Y_val)
)

# =========================
# 6. FINAL TEST (ONLY ONCE)
# =========================

loss, acc = model.evaluate(X_test, Y_test)
print("Final Test Accuracy:", acc)

# =========================
# 7. SAVE MODEL
# =========================

model.save("sax_model.h5")
print("Model saved!")

###PLOT KRIVE TRENINGA
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy po epohama')
plt.xlabel('Epoha')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss po epohama')
plt.xlabel('Epoha')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_krive.png', dpi=150)
plt.show()