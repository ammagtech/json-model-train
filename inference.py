def generate_output(input_data):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "output_model"  # Make sure it exists from training
    prompt = input_data.get("prompt", "Hello!")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(output[0], skip_special_tokens=True)

    return result
