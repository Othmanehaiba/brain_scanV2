from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os

# Configuration par défaut du DAG
default_args = {
    'owner': 'Othmane',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_data_availability():
    """Vérifie la présence des données brutes avant de lancer l'entraînement."""
    raw_path = "data/raw"
    if not os.path.exists(raw_path) or len(os.listdir(raw_path)) == 0:
        raise FileNotFoundError(f"Dossier {raw_path} introuvable ou vide.")
    print("✅ Données brutes validées !")


with DAG(
    'brain_tumor_mlops_pipeline',
    default_args=default_args,
    description='Pipeline MLOps automatisé pour l analyse d images cérébrales',
    schedule_interval='@monthly',  # Exécution automatique mensuelle
    catchup=False,
) as dag:

    # 1. Tâche de vérification
    task_check_data = PythonOperator(
        task_id='check_raw_data',
        python_callable=check_data_availability
    )

    # 2. Tâche de prétraitement classification
    task_prep_classif = BashOperator(
        task_id='prep_classification_data',
        bash_command='python src/data_prep/prep_classification.py'
    )

    # 3. Tâche de prétraitement détection
    task_prep_detect = BashOperator(
        task_id='prep_detection_data',
        bash_command='python src/data_prep/prep_detection.py'
    )

    # 4. Tâche de notification/validation finale
    def log_pipeline_success():
        print("🚀 Pipeline MLOps exécuté avec succès. Les modèles sont prêts pour la mise en production sur FastAPI.")

    task_notify = PythonOperator(
        task_id='notify_success',
        python_callable=log_pipeline_success
    )

    # Définition des dépendances (Ordre d'exécution)
    task_check_data >> [task_prep_classif, task_prep_detect] >> task_notify