import os
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Emplacements des données extraites
DATASET_DIR = "processed/classification" # ou "data/raw/classification"
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# 1. Chargement des jeux de données avec tf.data
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 2. Normalisation [-1, 1] pour InceptionV3
preprocess = tf.keras.applications.inception_v3.preprocess_input
train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

# 3. Construction du modèle (Feature Extraction)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# 4. Phase 1 : Feature Extraction (10 époques)
print(">>> Phase 1 : Feature Extraction...")
model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=20)

# 5. Phase 2 : Fine-Tuning (10 époques)
print("\n>>> Phase 2 : Fine-Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=20)

# 6. Sauvegarde du modèle
os.makedirs("models/classification", exist_ok=True)
model_path = "models/classification/inceptionv3_brain_tumor.keras"
model.save(model_path)
print(f"Modèle de classification sauvegardé dans {model_path}")

