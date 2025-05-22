import os
import torch
import re
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.cuda.amp import autocast
from value_head import ValueHead
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
import torch.nn.functional as F
import gc
import re

def get_quant_config(quantization: str | None):
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        return None

def cast_layernorm_and_embedding_to_half(model):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.Embedding)):
            module.to(torch.float16)

class CodeGenerator(nn.Module):
    DEFAULT_MODEL_NAME = "Qwen/Qwen1.5-0.5B"

    def __init__(
        self,
        load_path=None,
        model_name_or_path=None,
        device="cuda",
        max_length=1024,
        max_new_tokens=512,
        quantization="8bit",
        lora=True,
        attn_implementation="flash_attention_2",
    ):
        super().__init__()
        model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_NAME

        torch.set_float32_matmul_precision('high')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        if load_path and os.path.exists(os.path.join(load_path, "tokenizer")):
            tokenizer_path = os.path.join(load_path, "tokenizer")
        else:
            tokenizer_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # === Model Loading ===
        quant_config = get_quant_config(quantization)

        if load_path and os.path.exists(load_path):
            print(f"Loading model + LoRA from {load_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attn_implementation=attn_implementation
            )
            base_model = prepare_model_for_kbit_training(base_model)
            self.model = PeftModel.from_pretrained(base_model, load_path)
        else:
            print(f"Initializing new model from base: {model_name_or_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attn_implementation=attn_implementation
            )
            if lora:
                print("Applying LoRA adaptation...")
                self.model = prepare_model_for_kbit_training(self.model)
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, lora_config)

        cast_layernorm_and_embedding_to_half(self.model)
        self.model.half()
        self.model.config.torch_dtype = torch.float16
        self.enable_lora_and_top_blocks_gradients()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.gradient_checkpointing_disable()

        # === Value Head ===
        hidden_dim = self.model.config.hidden_size
        self.value_head = ValueHead(hidden_dim).to(self.device)

        vh_path = os.path.join(load_path or "", "value_head.pt")
        if os.path.exists(vh_path):
            self.value_head.load_state_dict(torch.load(vh_path))
            print("Loaded value head.")
        else:
            print("No value head to load")

        self.to(self.device)



    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA adapter
        if isinstance(self.model, PeftModel):
            print("Saving LoRA adapter...")
            self.model.save_pretrained(save_path)
        else:
            print("Warning: model is not a PeftModel, skipping LoRA adapter save.")

        # Save value head
        if hasattr(self, "value_head"):
            vh_path = os.path.join(save_path, "value_head.pt")
            torch.save(self.value_head.state_dict(), vh_path)
            print(f"Saved value head to {vh_path}")
        else:
            print("Warning: value head not found, skipping.")

        # Save tokenizer
        tokenizer_path = os.path.join(save_path, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def enable_lora_and_top_blocks_gradients(self):
        print(f"Freezing all parameters except LoRA blocks...")

        # Freeze all parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Enable LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Print summary
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")


    def create_code_fixer_prompt(self, bad_code):
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


    def create_short_prompt_header(self, requirement: str) -> str:
        return f"""You are an Unreal Engine C++ developer.

Your task is to implement ONLY the .h file code that corresponds to the given requirement.

Do NOT include any explanation, comments, or .cpp file.
Do NOT describe the requirement or restate it.
Write only the .h code inside a ```cpp code block.
Start your C++ code inside a ```cpp code block and close it properly with ```.
Always use UCLASS, USTRUCT.
Always start with #pragma once

---

The Requirement below is what you need to implement in Your Answer. **Answer Code Only**

Requirement:
{requirement}

Your Answer:
"""

    def create_short_prompt_cpp(self, requirement: str, header_code: str) -> str:
        return f"""You are an Unreal Engine C++ developer.

Your task is to implement ONLY the .cpp file code that corresponds to the given requirement and header.

Do NOT include any explanation, comments, or .h file.
Do NOT describe the requirement or restate it.
Write only the .cpp code inside a ```cpp code block.
Start your C++ code inside a ```cpp code block and close it properly with ```.

---

Requirement:
{requirement}

Corresponding Header File:
{header_code}

---

Now write ONLY the corresponding .cpp file code.
Your Answer:
"""


    def forward(self, input_ids, attention_mask=None, labels=None):
        self.model.train()
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)

    def sample_code_batch_with_partial_grad_caching(self, prompts: list[str], temperature=0.3, top_p=0.85, repetition_penalty=1.0, max_track_tokens=32):
        self.model.train()

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.size(0)
        eos_token_id = self.tokenizer.eos_token_id

        generated_token_ids = [[] for _ in range(B)]
        all_log_probs = []

        for i in range(B):  # 직렬화: 샘플별 처리
            cur_input_ids = input_ids[i:i+1]  # [1, T]
            cur_attention_mask = attention_mask[i:i+1]
            finished = False
            code_fence_count = 0
            backtick_count = 0
            past_key_values = None
            cur_log_probs = []

            track_flags = torch.zeros(self.max_new_tokens, dtype=torch.bool)

            if max_track_tokens == 0:
                track_flags[:] = False
            else:
                # 앞쪽 max_track_tokens 개는 반드시 추적
                track_flags[:max_track_tokens] = True

                # 그 이후의 토큰 중 일부도 랜덤 추적
                additional_track = torch.rand(self.max_new_tokens - max_track_tokens) < 0.1
                track_flags[max_track_tokens:] = additional_track


            for t in range(self.max_new_tokens):
                grad_tracking = track_flags[t].item()

                if t == 0:
                    inputs_embeds = self.model.get_input_embeddings()(cur_input_ids).to(torch.float16)
                else:
                    last_token_ids = cur_input_ids[:, -1:]
                    inputs_embeds = self.model.get_input_embeddings()(last_token_ids).to(torch.float16)


                inputs_embeds = inputs_embeds.to(self.model.dtype)

                if grad_tracking:
                    inputs_embeds.requires_grad_()
                    outputs = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=cur_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :] / temperature
                    logits = RepetitionPenaltyLogitsProcessor(repetition_penalty)(cur_input_ids, logits)
                    logits = TopPLogitsWarper(top_p)(cur_input_ids, logits)
                    probs = F.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1)
                    log_prob = torch.log(torch.gather(probs, dim=-1, index=sampled) + 1e-8)
                    cur_log_probs.append(log_prob.squeeze(1))
                else:
                    with torch.no_grad():
                        outputs = self.model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=cur_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                        past_key_values = outputs.past_key_values
                        logits = outputs.logits[:, -1, :] / temperature
                        logits = RepetitionPenaltyLogitsProcessor(repetition_penalty)(cur_input_ids, logits)
                        logits = TopPLogitsWarper(top_p)(cur_input_ids, logits)
                        probs = F.softmax(logits, dim=-1)
                        sampled = torch.multinomial(probs, num_samples=1)

                token_id = sampled[0].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                backtick_count += token_text.count("`")

                if not finished:
                    generated_token_ids[i].append(token_id)

                    # 종료 조건 1: EOS
                    if token_id == eos_token_id:
                        finished = True

                    # 종료 조건 2: 코드 fence 감지
                    elif token_text.strip() == "```":
                        code_fence_count += 1
                        if code_fence_count >= 2:
                            finished = True

                    # 종료 조건 3: 백틱 6개 이상
                    elif backtick_count >= 6:
                        finished = True


                cur_input_ids = torch.cat([cur_input_ids, sampled.detach()], dim=1)
                cur_attention_mask = torch.cat([cur_attention_mask, torch.ones_like(sampled)], dim=1)

                del outputs, logits, probs, sampled, inputs_embeds
                if 'log_prob' in locals():
                    del log_prob

                if finished:
                    break

            if cur_log_probs:
                all_log_probs.append(torch.stack(cur_log_probs))
            else:
                all_log_probs.append(torch.zeros(1, device=self.device))

        final_log_probs = torch.nn.utils.rnn.pad_sequence(all_log_probs, batch_first=True)  # [B, T']
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        return {
            "texts": generated_texts,
            "log_probs": final_log_probs,
        }

    def sample_header_with_partial_grad(self, requirements: list[str], max_track_tokens: int = 32) -> dict:
        result = {
            "header_texts": [],
            "header_log_probs": []
        }

        header_prompts = [self.create_short_prompt_header(req) for req in requirements]

        out = self.sample_code_batch_with_partial_grad_caching(
            header_prompts, max_track_tokens=max_track_tokens
        )

        result["header_texts"] = out["texts"]
        result["header_log_probs"] = out["log_probs"]

        del out
        torch.cuda.empty_cache()

        max_len = result["header_log_probs"].shape[1]
        result["header_log_probs"] = result["header_log_probs"].to(self.device)

        return result

    def sample_cpp_with_partial_grad(self, requirements: list[str], header_texts: list[str], max_track_tokens: int = 32) -> dict:
        assert len(requirements) == len(header_texts), "Length mismatch between requirements and headers"

        result = {
            "cpp_texts": [],
            "cpp_log_probs": []
        }

        cpp_prompts = [self.create_short_prompt_cpp(req, header) for req, header in zip(requirements, header_texts)]

        cpp_out = self.sample_code_batch_with_partial_grad_caching(
            cpp_prompts, max_track_tokens=max_track_tokens
        )

        result["cpp_texts"] = cpp_out["texts"]
        result["cpp_log_probs"] = cpp_out["log_probs"]

        del cpp_out
        torch.cuda.empty_cache()

        result["cpp_log_probs"] = result["cpp_log_probs"].to(self.device)

        return result

    def compute_value(self, prompts: list[str], responses: list[str], mode: str = "h") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.value_head.eval()

        full_inputs = [p + r for p, r in zip(prompts, responses)]

        inputs = self.value_head.tokenizer(
            full_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        value_total, value_disc, value_format = self.value_head(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            mode=mode
        )

        return value_total.squeeze(-1), value_disc.squeeze(-1), value_format.squeeze(-1)

    def generate(self, prompts: list[str] | str, return_text: bool = True) -> list[str] | dict[str, list[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]

        def extract_or_wrap_cpp_block(text: str) -> str:
            match = re.search(r"(```cpp\s*.*?\s*```)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return text.strip()

        # === HEADER ===
        header_prompts = [self.create_short_prompt_header(req) for req in prompts]
        header_inputs = self.tokenizer(header_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)

        with torch.no_grad():
            header_outputs = self.model.generate(
                **header_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.0,
                return_dict_in_generate=True
            )

        gen_start = header_inputs["input_ids"].shape[1]
        raw_header_texts = [self.tokenizer.decode(seq[gen_start:], skip_special_tokens=True) for seq in header_outputs.sequences]
        header_texts = [extract_or_wrap_cpp_block(text) for text in raw_header_texts]

        # === CPP ===
        cpp_prompts = [self.create_short_prompt_cpp(req, hdr) for req, hdr in zip(prompts, header_texts)]
        cpp_inputs = self.tokenizer(cpp_prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)

        with torch.no_grad():
            cpp_outputs = self.model.generate(
                **cpp_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.0,
                return_dict_in_generate=True
            )

        gen_start = cpp_inputs["input_ids"].shape[1]
        raw_cpp_texts = [self.tokenizer.decode(seq[gen_start:], skip_special_tokens=True) for seq in cpp_outputs.sequences]
        cpp_texts = [extract_or_wrap_cpp_block(text) for text in raw_cpp_texts]

        # === 결과 정리 ===
        if return_text:
            return [h + "\n" + c for h, c in zip(header_texts, cpp_texts)]
        else:
            return {
                "header_texts": header_texts,
                "cpp_texts": cpp_texts
            }


# Main
if __name__ == "__main__":
    generator = CodeGenerator(load_path="./checkpoint_1_3b/generator", attn_implementation="eager")
    prompts = [
        "Create a character class with health and mana properties.",
        "Create an enemy class that attacks the player and can die."
    ]

    result = generator.generate(prompts, return_text=False)

    for i, (h, c) in enumerate(zip(result["header_texts"], result["cpp_texts"])):
        print(f"\n=== Sample {i+1} Header ===\n{h}")
        print(f"=== Sample {i+1} CPP ===\n{c}")
