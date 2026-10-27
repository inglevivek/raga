# pyrefly: ignore [missing-import]
from datasets import load_dataset
from model_loader import load_model
# pyrefly: ignore [missing-import]
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

model, tokenizer = load_model()

dataset = load_dataset("tatsu-lab/alpaca", split= "train").select(range(1000))
print(dataset)

def tokenize(batch):
    return tokenizer(
        batch["instruction"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

dataset= dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset= dataset.train_test_split(test_size=0.2)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir= "ragarag",
    per_device_train_batch_size= 2,
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing= True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps= 10,
    save_steps= 10,
    eval_strategy= "epoch",
    save_strategy="epoch",
    load_best_model_at_end= True,
    report_to= "none"
)

trainer = Trainer(
    model= model,
    args= training_args,
    train_dataset= dataset["train"],
    eval_dataset= dataset["test"],
    processing_class=tokenizer,
    data_collator= data_collator
)

trainer.train()
trainer.save_model("ragarag_finetuned")