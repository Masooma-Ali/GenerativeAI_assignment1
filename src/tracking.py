import contextlib, os
from pathlib import Path


def mlflow_run(enabled, experiment, run_name, base_dir):
    """Context manager: MLflow run on a SQLite backend (the old file store is disabled in new MLflow)."""
    if not enabled:
        return contextlib.nullcontext()
    import mlflow
    base = os.path.abspath(base_dir); os.makedirs(base, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base, 'mlflow.db')}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=Path(os.path.join(base, "artifacts")).as_uri())
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
