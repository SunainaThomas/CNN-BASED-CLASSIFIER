# PROJECT: Object Recognition in Images
# CNN-based classifier trained on the CIFAR-10 dataset

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

# The 10 categories our model will learn to tell apart
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES  = 10

# Step 1 – Load and prepare the data

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"  Training images : {x_train.shape}")
print(f"  Test images     : {x_test.shape}")
print(f"  Classes         : {CLASS_NAMES}\n")

# Step 2 – Data augmentation

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  
    layers.RandomRotation(0.1),        
    layers.RandomZoom(0.1),            
], name="augmentation")

# Step 3 – Build the CNN
# ─────────────────────────────────────────────
# Stacked three convolutional blocks. Each block:
#   • Two Conv2D layers  → learn visual patterns (edges → shapes → objects)
#   • BatchNormalization → keeps activations well-scaled during training
#   • MaxPooling         → halves the spatial size, reduces computation
#   • Dropout            → randomly zeroes some neurons to prevent overfitting

def build_cnn():
    inputs = tf.keras.Input(shape=(32, 32, 3), name="image_input")
    x = augmentation(inputs)

    # ── Block 1: detect low-level features (edges, colours) 
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  
    x = layers.Dropout(0.25)(x)

    # ── Block 2: detect mid-level features (parts of objects) 
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)   
    x = layers.Dropout(0.25)(x)

    # ── Block 3: detect high-level features (whole objects) ──
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)  
    x = layers.Dropout(0.25)(x)

    # ── Classifier head ──
    x = layers.Flatten()(x)                  
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)               
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="CIFAR10_CNN")


model = build_cnn()
model.summary()



# Step 4 – Compile and train

# Adam is a solid all-round optimiser. sparse_categorical_crossentropy
# is the standard loss for multi-class problems with integer labals.
# Two callbacks keep training smart:
#   EarlyStopping  – stops when validation loss stops improving (saves time)
#   ReduceLROnPlateau – lowers the learning rate when we hit a plateau

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

print("\nStarting training (up to 50 epochs, early stopping enabled)...")
history = model.fit(
    x_train, y_train,
    validation_split=0.1,   # hold out 10% of training data to monitor overfitting
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# Step 5 – Evaluate on the test set
print("\nEvaluating on 10,000 test images...")
y_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)   

cm = confusion_matrix(y_test, y_pred)
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (TP + FP + FN)
eps = 1e-9
precision_per_class   = TP / (TP + FP + eps)
recall_per_class      = TP / (TP + FN + eps)    
specificity_per_class = TN / (TN + FP + eps)
f1_per_class          = (2 * precision_per_class * recall_per_class /
                         (precision_per_class + recall_per_class + eps))

accuracy    = accuracy_score(y_test, y_pred)
precision_m = precision_score(y_test, y_pred, average='macro')
recall_m    = recall_score(y_test, y_pred, average='macro')
f1_m        = f1_score(y_test, y_pred, average='macro')
sensitivity = recall_m                        
specificity = specificity_per_class.mean()   

print("\n" + "=" * 55)
print("  OVERALL PERFORMANCE ON TEST SET")
print("=" * 55)
print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision    : {precision_m:.4f}  (macro average)")
print(f"  Recall       : {recall_m:.4f}  (macro average)")
print(f"  Sensitivity  : {sensitivity:.4f}  (same as Recall)")
print(f"  Specificity  : {specificity:.4f}  (macro average)")
print(f"  F1-Score     : {f1_m:.4f}  (macro average)")
print("=" * 55)
print("\nDetailed per-class breakdown:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
# Step 6 – Visualisations
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(history.history['accuracy'],     label='Training',   linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy over Epochs')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],     label='Training',   linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
axes[1].set_title('Model Loss over Epochs')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=120)
plt.show()
print("Saved: training_curves.png")
plt.figure(figsize=(11, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5)
plt.title('Confusion Matrix\n(rows = actual class, columns = predicted class)', pad=12)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.show()
print("Saved: confusion_matrix.png")
x_pos = np.arange(NUM_CLASSES)
bar_width = 0.2

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x_pos - 1.5*bar_width, precision_per_class,   bar_width, label='Precision',            alpha=0.85)
ax.bar(x_pos - 0.5*bar_width, recall_per_class,      bar_width, label='Recall / Sensitivity', alpha=0.85)
ax.bar(x_pos + 0.5*bar_width, specificity_per_class, bar_width, label='Specificity',          alpha=0.85)
ax.bar(x_pos + 1.5*bar_width, f1_per_class,          bar_width, label='F1-Score',             alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score (0 – 1)')
ax.set_title('Per-class Metrics Comparison')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=120)
plt.show()
print("Saved: per_class_metrics.png")
fig, axes = plt.subplots(3, 6, figsize=(14, 7))
sample_indices = np.random.choice(len(x_test), 18, replace=False)

for ax, idx in zip(axes.flat, sample_indices):
    ax.imshow(x_test[idx])
    true_label = CLASS_NAMES[y_test[idx]]
    pred_label = CLASS_NAMES[y_pred[idx]]
    confidence = y_prob[idx][y_pred[idx]] * 100   # model's confidence in %
    correct = (true_label == pred_label)
    color = 'green' if correct else 'red'
    ax.set_title(f"True : {true_label}\nPred : {pred_label} ({confidence:.0f}%)",
                 color=color, fontsize=7)
    ax.axis('off')

plt.suptitle("Sample Predictions — green = correct, red = wrong", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=120)
plt.show()
print("Saved: sample_predictions.png")

print("\nDONE")
