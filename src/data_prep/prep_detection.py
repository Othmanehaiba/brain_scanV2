import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import glob

PROCESSED_YOLO_DIR = Path("data/processed/detection")
CLASS_MAPPING = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
RAW_DATA_DIR = Path("data/raw")



def validate_yolo_annotation(txt_path: Path, num_classes=4) -> bool:
    """Vérifie le format et la validité des coordonnées YOLO."""
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return False
    
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return False
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Vérification des plages [0, 1] et identifiant de classe
                if cls_id < 0 or cls_id >= num_classes:
                    return False
                if any(c < 0.0 or c > 1.0 for c in coords):
                    return False
        return True
    except Exception:
        return False


def collect_matched_pairs(raw_dir: Path):
    """Associe chaque image à son fichier .txt d'annotation correspondant."""
    matched_pairs = [] # (image_path, label_path)
    
    # Recherche de toutes les images
    all_images = glob.glob(str(raw_dir / "**" / "images" / "**" / "*.[jp][pn]g"), recursive=True)
    
    orphan_images = 0
    orphan_labels = 0

    for img_str in all_images:
        img_path = Path(img_str)
        # Reconstitution du chemin du fichier label correspondant
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")

        if label_path.exists():
            if validate_yolo_annotation(label_path):
                matched_pairs.append((img_path, label_path))
            else:
                orphan_images += 1
        else:
            orphan_images += 1

    print(f"Paires (Image, Annotation) valides trouvées: {len(matched_pairs)}")
    print(f"Images ou annotations rejetées (invalides/orphelines): {orphan_images}")
    return matched_pairs


# 1. Récupération et filtrage des paires
pairs = collect_matched_pairs(RAW_DATA_DIR)

# 2. Répartition Train (70%), Val (15%), Test (15%)
train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=42)
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=42)


def build_yolo_structure(pairs_list, split_name):
    """Copie les paires images/labels dans l'arborescence YOLOv8."""
    img_dst_dir = PROCESSED_YOLO_DIR / "images" / split_name
    lbl_dst_dir = PROCESSED_YOLO_DIR / "labels" / split_name
    
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    for img_p, lbl_p in pairs_list:
        shutil.copy(img_p, img_dst_dir / img_p.name)
        shutil.copy(lbl_p, lbl_dst_dir / lbl_p.name)


# Copie des fichiers
build_yolo_structure(train_pairs, "train")
build_yolo_structure(val_pairs, "val")
build_yolo_structure(test_pairs, "test")

# 3. Génération du fichier configuration YOLO (data.yaml)
data_yaml_content = {
    "path": str(PROCESSED_YOLO_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {v: k for k, v in CLASS_MAPPING.items()}
}

config_file_path = Path("config/data.yaml")
os.makedirs("config", exist_ok=True)

with open(config_file_path, "w") as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"Fichier de configuration YOLO généré avec succès : {config_file_path}")