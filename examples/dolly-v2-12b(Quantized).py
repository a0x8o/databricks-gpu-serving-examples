# Databricks notebook source
!pip install --upgrade torch
!pip install accelerate
!pip install bitsandbytes
!pip install pynvml
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch

# COMMAND ----------

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

print_gpu_utilization()

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the Dolly model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="databricks/dolly-v2-12b")

# COMMAND ----------

class Dolly(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model

        quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], 
            padding_side="left")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            quantization_config=quantization_config,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        message = model_input["message"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Build the prompt
        prompt = self._build_prompt(message)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_length=max_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "message"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "message":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Dolly(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate","bitsandbytes"],
        input_example=input_example,
        signature=signature,
    )


# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "dolly-v2-12b"
)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
print_gpu_utilization()

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"message":["what is ML?"], "temperature": [0.5],"max_tokens": [100]})
loaded_model.predict(input_example)

# COMMAND ----------


