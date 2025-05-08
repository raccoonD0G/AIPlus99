import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit
from transformers import AutoTokenizer
from code_formatting_pairs_dataset import get_train_dataloader
from code_fixer_1_3b import CodeFixer

# Settings
train_data_path = "code_formatting_pairs.jsonl"
num_epochs = 300
batch_size = 5
learning_rate = 2e-5
save_path = "./checkpoint_1_3b"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
def train(model, dataloader, num_epochs=3, lr=2e-5, max_length=512, save_path="./checkpoint_1_3b"):
    optimizer = AdamW8bit(model.parameters(), lr=lr)
    model.train()

    IGNORE_INDEX = -100
    os.makedirs(save_path, exist_ok=True)
    tokenizer = model.tokenizer

    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)

        for batch in loop:
            bad_codes = batch["bad_code"]
            good_codes = batch["good_code"]

            prompts = [model.create_prompt(bad) for bad in bad_codes]
            answers = [f"```cpp\n{good}\n```" for good in good_codes]
            combined = [p + a for p, a in zip(prompts, answers)]

            tokenized = tokenizer(
                combined,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            labels = input_ids.clone()
            for i in range(len(prompts)):
                prompt_tokenized = tokenizer(prompts[i], truncation=True, max_length=max_length, return_tensors="pt").to(device)
                prompt_len = prompt_tokenized.input_ids.size(1)
                labels[i, :prompt_len] = IGNORE_INDEX

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        # Show example at the end of each epoch
        example_batch = next(iter(dataloader))
        example_bad_code = example_batch["bad_code"][0]
        model.generate_and_print_example(example_bad_code)

        # Save model
        model.model.save_pretrained(save_path)
        model.tokenizer.save_pretrained(save_path)

# Main
if __name__ == "__main__":
    model = CodeFixer()
    train_loader = get_train_dataloader(file_path=train_data_path, batch_size=batch_size, shuffle=True)
    train(model, train_loader, num_epochs=num_epochs, lr=learning_rate, save_path=save_path)
