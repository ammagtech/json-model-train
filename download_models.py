from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading Qwen2.5 model and tokenizer...")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
print("Download complete.")
