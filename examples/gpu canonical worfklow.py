# Databricks notebook source
#install mlflow from master branch
dbutils.fs.cp("dbfs:/FileStore/mlflow-2.3.3.dev0-py3-none-any.whl", "file:/tmp/mlflow-2.3.3.dev0-py3-none-any.whl")
!pip install /tmp/mlflow-2.3.3.dev0-py3-none-any.whl
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from transformers import pipeline
import numpy as np
import pandas as pd

# COMMAND ----------

sentiment_pipeline = pipeline("text-classification", device=0)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=sentiment_pipeline,
        artifact_path="sentiment",
        input_example="This restaurant is awesome",
        pip_requirements=(
            ["/tmp/mlflow-2.3.3.dev0-py3-none-any.whl"]
        ),
    )

# COMMAND ----------

import mlflow.models.utils
mlflow.models.utils.add_libraries_to_model(f"models:/gpt2-pipeline/latest")

# COMMAND ----------

# Load as interactive pyfunc
sentiment = mlflow.pyfunc.load_model(model_info.model_uri)
sentiment.predict(pd.DataFrame(["This restaurant is awesome"]))
