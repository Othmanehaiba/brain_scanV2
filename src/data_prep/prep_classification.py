import os
import shutil
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split

# --- Configuration des chemins ---
RAW_DATA_DIR = Path("data/raw")
PROCESSED_CLASSIF_DIR = Path("data/processed/classification")
CLASSES = ["glioma", "meningioma", "no tumor", "pituitary"]


def verify_image_integrity(image_path: Path) -> bool:
    """Vérifie l'extension et l'intégrité d'une image."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    if image_path.suffix.lower() not in valid_extensions:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Test supplémentaire de lecture OpenCV
        mat = cv2.imread(str(image_path))
        if mat is None:
            return False
        return True
    except Exception:
        return False


def collect_valid_images(raw_dir: Path):
    """Parcourt les sous-dossiers et écarte les fichiers invalides."""
    data = []
    invalid_count = 0

    for cls_name in CLASSES:
        # Recherche toutes les images dans les sous-dossiers
        images = glob.glob(str(raw_dir / "**" / cls_name / "**" / "*.jpg"), recursive=True) + \
                 glob.glob(str(raw_dir / "**" / cls_name / "**" / "*.png"), recursive=True)
        
        for img_p in images:
            p = Path(img_p)
            # Ne prendre en compte que les fichiers du dossier /images/ s'il existe
            if "labels" in p.parts:
                continue
            if verify_image_integrity(p):
                data.append({"filepath": str(p), "label": cls_name})
            else:
                invalid_count += 1

    print(f"Total d'images valides trouvées: {len(data)}")
    print(f"Fichiers corrompus ou invalides écartés: {invalid_count}")
    return pd.DataFrame(data)


# 1. Collecte et Intégrité
df_images = collect_valid_images(RAW_DATA_DIR)

# 2. Exploration et Affichage (Countplot)
plt.figure(figsize=(8, 5))
sns.countplot(data=df_images, x="label", palette="viridis")
plt.title("Distribution des images par classe")
plt.xlabel("Classes de tumeurs")
plt.ylabel("Nombre d'images")
plt.tight_layout()
plt.show()

# 3. Répartition Train (70%), Validation (15%), Test (15%)
train_df, temp_df = train_test_split(
    df_images, test_size=0.30, stratify=df_images["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

print(f"\n--- Répartition du Dataset ---")
print(f"Train: {len(train_df)} ({len(train_df)/len(df_images):.1%})")
print(f"Validation: {len(val_df)} ({len(val_df)/len(df_images):.1%})")
print(f"Test: {len(test_df)} ({len(test_df)/len(df_images):.1%})")


# 4. Pipeline d'Augmentation de données avec Albumentations
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomBrightnessContrast(p=0.2)
])


def copy_and_augment(df, split_name, augment=False, factor=1):
    """Copie les images dans leur dossier final et applique l'augmentation sur le Train."""
    for cls in CLASSES:
        os.makedirs(PROCESSED_CLASSIF_DIR / split_name / cls, exist_ok=True)

    for _, row in df.iterrows():
        src_path = Path(row["filepath"])
        cls = row["label"]
        dst_path = PROCESSED_CLASSIF_DIR / split_name / cls / src_path.name
        
        # Copie simple
        shutil.copy(src_path, dst_path)
        # Augmentation réservée uniquement à l'ensemble d'entraînement
        if augment:
            img = cv2.imread(str(src_path))
            for i in range(factor):
                augmented = augmentation_pipeline(image=img)["image"]
                aug_name = f"{src_path.stem}_aug_{i}{src_path.suffix}"
                aug_dst_path = PROCESSED_CLASSIF_DIR / split_name / cls / aug_name
                cv2.imwrite(str(aug_dst_path), augmented)


# Export des images
print("\nExportation et augmentation des données en cours...")
copy_and_augment(train_df, "train", augment=True, factor=1) # Génère +1 copie augmentée par image train
copy_and_augment(val_df, "val", augment=False)
copy_and_augment(test_df, "test", augment=False)
print("Préparation de la classification terminée !")