import os
import torch
import re
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class CodeFixer(nn.Module):
    DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

    def __init__(
        self,
        load_path=None,
        model_name_or_path=None,
        device="cuda",
        max_length=512,
        max_new_tokens=256,
        quantization="4bit",
        lora=True
    ):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        if load_path and os.path.exists(load_path):
            print(f"Loading model from checkpoint: {load_path}")
            model_name_or_path = load_path
            lora = False  # Prevent double wrapping
        else:
            model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_NAME
            print(f"Initializing new model from base: {model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto" if device is None else device,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if lora:
            print("Applying LoRA adaptation...")
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.gradient_checkpointing_disable()

        self.to(self.device)


    def create_prompt(self, bad_code):
        return f"""You are a code style expert specialized in Unreal Engine C++.

You must strictly follow the official style guide, including:
- Always open braces {{}} on a new line.
- Group access specifiers (public, private, protected) properly.
- Use UPROPERTY() and UFUNCTION() macros correctly, one variable or function per line.
- Use PascalCase naming for variables and functions.
- Separate #include statements by engine and project, and add blank lines between groups.
- Insert line breaks after semicolons for better readability.

Now, fix the following BAD_CODE to fully comply with the style guide.
Return only the corrected GOOD_CODE without any explanation.
Once you finish writing GOOD_CODE, close the code block by typing ``` and immediately stop generating further text.

BAD_CODE:
```cpp
{bad_code}
```
GOOD_CODE:
"""

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def fix_code(self, bad_code: str) -> str:
        self.model.eval()

        prompt = self.create_prompt(bad_code)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        output_tokens = outputs[0][prompt_length:]

        if self.tokenizer.eos_token_id in output_tokens:
            eos_idx = (output_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
            output_tokens = output_tokens[:eos_idx]

        predicted_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

        if "```" in predicted_text:
            fixed_code = re.split(r"```[a-zA-Z]*\n?", predicted_text, maxsplit=1)[-1]
            fixed_code = fixed_code.split("```")[0].strip()
        else:
            fixed_code = predicted_text.strip()

        return fixed_code

    def generate_and_print_example(self, bad_code: str):
        self.model.eval()
        fixed_code = self.fix_code(bad_code)

        print("\n[Example Code]")
        print(bad_code)
        print("[Example Prediction]")
        print(f"{fixed_code}")

        self.model.train()


# Main
if __name__ == "__main__":
    load_path = "./checkpoint_1_3b"
    model = CodeFixer()
    model.generate_and_print_example("if(\nHealth<=0)Destroy();");