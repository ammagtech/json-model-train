import runpod
import torch
import os

# Optional: suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load training and inference modules
from train import train_model
from inference import generate_output  # You create this function

def handler(event):
    input_data = event.get('input', {})
    mode = input_data.get("mode", "inference")

    if mode == "train":
        print("Training requested...")
        output = train_model()  # Trains and returns logs or success flag
    else:
        print("Inference requested...")
        output = generate_output(input_data)  # Inference on input prompt

    return {
        "status": "success",
        "mode": mode,
        "output": output
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
