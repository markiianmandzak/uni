import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer
)

# --- Configuration ---
MODEL_NAME = "deepset/gelectra-base-germanquad"
OUTPUT_DIR = "./tier2_qa_model"
BATCH_SIZE = 16
NUM_EPOCHS = 2

def prepare_train_features(examples, tokenizer):
    """
    Tokenizes the text and finds the token-level start and end positions 
    of the answers.
    """
    # Tokenize questions and contexts
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second", 
        max_length=256,       # <--- REDUCED from 384 (Massive compute saving)
        stride=64,            # <--- REDUCED from 128
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one long context might give multiple features, we need a map
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answer is provided, point to the CLS token
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the start and end of the context in the token sequence
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (if we truncated)
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Move token_start_index and token_end_index to the answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

def main():
    print("1. Loading raw dataset...")
    # Load the JSONL files we just created
    dataset = load_dataset("json", data_files={
        "train": "train_qa_dataset.jsonl",
        "validation": "val_qa_dataset.jsonl"
    })

    # HACKATHON SPEED TRICK: 
    # Let's train on 50,000 rows and validate on 5,000 to see fast results.
    # Comment these two lines out later to train on the full 1.6M dataset!
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(50000))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(5000))

    print("2. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("3. Preprocessing and Tokenizing (This might take a minute)...")
    tokenized_datasets = dataset.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print("4. Loading Model...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Configure training arguments (Optimized for speed on M-series Mac / Single GPU)
# Configure training arguments (M3 MAX HYPER-SPEED)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",              # <--- Skip eval during training to save time
        learning_rate=3e-5,
        per_device_train_batch_size=32,  # <--- Drop to 32 (Prevents MPS throttling)
        gradient_accumulation_steps=2,   # <--- Keeps effective batch size at 64
        num_train_epochs=1,              # <--- Just 1 Epoch for a fast baseline!
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        dataloader_num_workers=4,        # <--- 4 is the sweet spot for Apple Silicon
        dataloader_pin_memory=False,
        bf16=True,                       # <--- THE MAGIC: M3 BFloat16 hardware acceleration
        # fp16=True,                     # (If bf16 throws an error, comment it out and uncomment fp16)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
    )

    print("5. Starting Training...")
    trainer.train()

    print("6. Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()