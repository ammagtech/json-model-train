def train_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import load_dataset
    import torch

    model_name = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()

    # Load your dataset - using .jsonl as you specified
    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

    def preprocess(examples):
        # Concatenate 'input' and 'output' to form the full sequence for the model to learn
        # Add a newline for separation and the tokenizer's EOS token at the end
        full_text = f"Input: {examples['input']}\nOutput: {examples['output']}{tokenizer.eos_token}"

        # Tokenize the combined text
        tokenized_inputs = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # Crucially: For causal language modeling, the labels are the input_ids themselves.
        # The Trainer will automatically shift these labels to compute the next-token prediction loss.
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

        return tokenized_inputs

    tokenized = train_dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="output_model",
        per_device_train_batch_size=1, # Keep this low for memory, adjust if you have more VRAM
        num_train_epochs=1,
        save_strategy="epoch",
        logging_dir="logs",
        # Consider adding logging_steps for more frequent updates, e.g., logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer, # It's good practice to pass the tokenizer here
    )

    print("Starting training...")
    trainer.train()
    print("Training finished. Saving model...")

    trainer.save_model("output_model")
    tokenizer.save_pretrained("output_model")
    print("Model and tokenizer saved to 'output_model'.")

    return "Training completed and model saved."