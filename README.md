# 🧠 BrainScan AI: Unified Brain Tumor Classification & Detection

Solution complète MLOps d'analyse d'images IRM cérébrales combinant la classification globale par **InceptionV3** et la détection/localisation précise par **YOLOv8**.

## 🚀 Architecture & Fonctionnalités
- **Classification :** InceptionV3 pré-entraîné (Transfer Learning + Fine-Tuning) sur 4 classes (*Glioma, Meningioma, NoTumor, Pituitary*).
- **Détection :** YOLOv8 Nano ($\text{mAP}@0.5 = 96.1\%$) pour la localisation par boîtes englobantes.
- **Backend API :** Endpoints REST rapides construits avec **FastAPI**.
- **Interface Web :** Dashboard interactif développé sous **Streamlit**.
- **Orchestration :** DAG **Apache Airflow** pour l'automatisation du pipeline de données.
- **Conteneurisation :** Déploiement multi-services via **Docker Compose**.

## 🛠️ Démarrage Rapide (Docker)

```bash
docker compose up --build