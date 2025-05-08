import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit
from code_generator_1_3b import CodeGenerator
from requirement_to_code_dataset import get_train_dataloader

# Settings
train_data_path = "requirement_to_code.jsonl"
num_epochs = 300
batch_size = 5
learning_rate = 2e-6
save_path = "./checkpoint_1_3b"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
def train(model, dataloader, num_epochs=3, lr=2e-5, max_length=2048, save_path="./checkpoint_1_3b"):
    optimizer = AdamW8bit(model.parameters(), lr=lr)
    model.train()

    IGNORE_INDEX = -100
    os.makedirs(save_path, exist_ok=True)
    tokenizer = model.tokenizer

    example_batch = next(iter(dataloader))
    example_requirement = example_batch["requirement"][0]
    model.generate_and_print_example(example_requirement)
    
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)

        for batch in loop:
            requirements = batch["requirement"]
            generated_codes = batch["generated_code"]

            prompts = [model.create_prompt(req) for req in requirements]
            answers = [f"```cpp\n{code}\n```" for code in generated_codes]
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
                prompt_len = prompt_tokenized.input_ids.shape[1]
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

        example_batch = next(iter(dataloader))
        example_requirement = example_batch["requirement"][0]
        model.generate_and_print_example(example_requirement)

        # Save model
        model.model.save_pretrained(save_path)
        model.tokenizer.save_pretrained(save_path)

# Main
if __name__ == "__main__":
    model = CodeGenerator.load_or_initialize(save_path)
    train_loader = get_train_dataloader(file_path=train_data_path, batch_size=batch_size, shuffle=True)
    train(model, train_loader, num_epochs=num_epochs, lr=learning_rate, save_path=save_path)
