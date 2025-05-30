def train_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import load_dataset
    import torch

    model_name = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()

    # Load your real JSON dataset
    train_dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

    def preprocess(examples):
        # *** CHANGE THIS LINE ***
        # Use 'input' instead of 'prompt' to match your dataset structure
        return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)

    tokenized = train_dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="output_model",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_dir="logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished. Saving model...")
    trainer.save_model("output_model")
    tokenizer.save_pretrained("output_model")
    print("Model and tokenizer saved to 'output_model'.")

    return "Training completed and model saved."