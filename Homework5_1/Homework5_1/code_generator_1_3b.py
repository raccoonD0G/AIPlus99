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

def print_model_structure(model, indent=0, max_depth=3):
    if indent > max_depth:
        return
    prefix = "  " * indent
    print(f"{prefix}{type(model)}")
    for name, child in model.named_children():
        print(f"{prefix}- {name}: {type(child)}")
        print_model_structure(child, indent + 1, max_depth)

class CodeGenerator(nn.Module):
    DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

    def __init__(
        self,
        load_path=None,
        model_name_or_path=None,
        device="cuda",
        max_length=1024,
        max_new_tokens=512,
        quantization="4bit",
        lora=True,
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
            tokenizer_path = model_name_or_path or self.DEFAULT_MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # === Model Loading ===
        if load_path and os.path.exists(load_path):
            print(f"Loading model + LoRA from {load_path}")
            if quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quant_config = None

            base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            base_model = prepare_model_for_kbit_training(base_model)
            self.model = PeftModel.from_pretrained(base_model, load_path)
        else:
            print(f"Initializing new model from base: {model_name_or_path}")
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
                device_map="auto",
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
                    target_modules = ["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(self.model, lora_config)

        self.enable_lora_and_top_blocks_gradients()
        self.model.gradient_checkpointing_enable()

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


    def create_prompt_header(self, requirement: str) -> str:
        return f"""You are an Unreal Engine C++ developer.

Write only the **Unreal Engine C++ header file (.h)** that satisfies the following requirement.
Start your C++ code inside a ```cpp code block and close it properly with ```.
Do not include any implementation or .cpp code.

Always start with a header file declaration like:
```cpp
// MovingActor.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MovingActor.generated.h"

Strict mode: Output only raw C++ code. Any non-code output is considered a mistake.

Here is an example.

Requirement:
Create an actor that moves forward constantly

Your Answer:
```cpp
// MovingActor.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MovingActor.generated.h"

UCLASS()
class MYPROJECT_API AMovingActor : public AActor
{{
    GENERATED_BODY()

    public:
    AMovingActor();

protected:
    virtual void BeginPlay() override;

public:
    virtual void Tick(float DeltaTime) override;

private:
    UPROPERTY(EditAnywhere)
    FVector MovementSpeed;
}};
```

The Requirement below is what you need to implement in Your Answer. **Answer Code Only**

Requirement:
{requirement}

Your Answer:
"""

    def create_prompt_cpp(self, requirement: str, header_code: str) -> str:
        return f"""You are an Unreal Engine C++ developer.

Write only the corresponding **Unreal Engine .cpp file** implementation.
Start your C++ code inside a ```cpp code block and close it properly with ```.

Always start with a header file declaration like:
```cpp
// MovingActor.cpp
#include "MovingActor.h"

AMovingActor::AMovingActor()
{{

Strict mode: Output only raw C++ code. Any non-code output is considered a mistake.

Here is an example.

Requirement:
Create an actor that moves forward constantly

Your Answer:
```cpp
// MovingActor.cpp
#include "MovingActor.h"

AMovingActor::AMovingActor()
{{
    PrimaryActorTick.bCanEverTick = true;
    MovementSpeed = FVector(100.f, 0.f, 0.f);
}}

void AMovingActor::BeginPlay()
{{
    Super::BeginPlay();
}}

void AMovingActor::Tick(float DeltaTime)
{{
    Super::Tick(DeltaTime);
    AddActorLocalOffset(MovementSpeed * DeltaTime);
}}
```

The Requirement below is what you need to implement in Your Answer. The Header File below is the header file (.h) that satisfies the requirement. Implement this Header File into Your Answer(.cpp).
**Answer Code Only**

Requirement:
{requirement}

Header File:
{header_code}

Your Answer:
"""

    def create_short_prompt_header(self, requirement: str) -> str:
        return f"""You are an Unreal Engine C++ developer.

Write only the **Unreal Engine C++ header file (.h)** that satisfies the following requirement.
Start your C++ code inside a ```cpp code block and close it properly with ```.

The Requirement below is what you need to implement in Your Answer. **Answer Header Code Only**

Requirement:
{requirement}

Your Answer:
"""

    def create_short_prompt_cpp(self, requirement: str, header_code: str) -> str:
        return f"""You are an Unreal Engine C++ developer.

Write only the corresponding **Unreal Engine .cpp file** implementation.
Start your C++ code inside a ```cpp code block and close it properly with ```.

The Requirement below is what you need to implement in Your Answer. **Answer CPP Code Only**

Requirement:
{requirement}

Header File:
{header_code}

Your Answer:
"""

    def forward(self, input_ids, attention_mask=None, labels=None):
        self.model.train()
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)

    def generate_header_and_cpp(self, requirement: str) -> tuple[str, str]:
        self.eval()

        # 헤더 코드 생성
        header_prompt = self.create_prompt_header(requirement)
        header_inputs = self.tokenizer(
            header_prompt, return_tensors="pt", truncation=True, padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.inference_mode():
            header_output = self.model.generate(
                **header_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        header_prompt_ids = self.tokenizer(header_prompt, return_tensors="pt").input_ids[0].to(self.device)
        header_generated = header_output[0][len(header_prompt_ids):]
        if self.tokenizer.eos_token_id in header_generated:
            eos_idx = (header_generated == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
            header_generated = header_generated[:eos_idx]
        header_code = self.tokenizer.decode(header_generated, skip_special_tokens=True).strip()

        # CPP 코드 생성
        cpp_prompt = self.create_prompt_cpp(requirement, header_code)
        cpp_inputs = self.tokenizer(
            cpp_prompt, return_tensors="pt", truncation=True, padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.inference_mode():
            cpp_output = self.model.generate(
                **cpp_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        cpp_prompt_ids = self.tokenizer(cpp_prompt, return_tensors="pt").input_ids[0].to(self.device)
        cpp_generated = cpp_output[0][len(cpp_prompt_ids):]
        if self.tokenizer.eos_token_id in cpp_generated:
            eos_idx = (cpp_generated == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
            cpp_generated = cpp_generated[:eos_idx]
        cpp_code = self.tokenizer.decode(cpp_generated, skip_special_tokens=True).strip()

        return header_code, cpp_code


    def generate_and_print_example(self, requirement: str):
        header, cpp = self.generate_header_and_cpp(requirement)

        print("\n[Requirement]")
        print(requirement)

        print("\n[Generated Header (.h)]")
        print(header)

        print("\n[Generated CPP (.cpp)]")
        print(cpp)


    def compute_random_start_idx(self, prompt_len: int, output_len: int, max_track_tokens: int, max_graph_tokens: int = 131072) -> int:
        base = max_track_tokens * prompt_len
        triangle = max_track_tokens * (max_track_tokens - 1) // 2
        remaining = max_graph_tokens - base - triangle

        vram_limited_start = max(0, remaining // max_track_tokens)

        output_limited_start = max(0, output_len - max_track_tokens)

        max_valid_start = min(vram_limited_start, output_limited_start)

        return torch.randint(0, max_valid_start + 1, (1,)).item()
    

    def sample_code_batch_with_partial_grad(self, prompts: list[str], temperature=0.7, top_p=0.9, repetition_penalty=1.1, max_track_tokens=30):
        self.model.train()

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        B = input_ids.size(0)

        eos_token_id = self.tokenizer.eos_token_id

        generated_token_ids = [[] for _ in range(B)]
        log_probs = []
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        code_fence_counts = [0] * B

        logits_processors = LogitsProcessorList([
            RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
            TopPLogitsWarper(top_p=top_p),
        ])

        current_input_ids = input_ids.detach().clone()
        current_attention_mask = attention_mask.detach().clone()

        # 무작위 추적 구간 선택
        start_idx = self.compute_random_start_idx(
            prompt_len=input_ids.shape[1],
            output_len=self.max_new_tokens,
            max_track_tokens=max_track_tokens,
            max_graph_tokens=524288
        )

        track_range = range(start_idx, start_idx + max_track_tokens)

        for t in range(self.max_new_tokens):
            grad_tracking = t in track_range

            if grad_tracking:
                inputs_embeds = self.model.get_input_embeddings()(current_input_ids)
                inputs_embeds.requires_grad_()
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=current_attention_mask,
                    use_cache=False,
                )
            else:
                with torch.no_grad():
                    inputs_embeds = self.model.get_input_embeddings()(current_input_ids)
                    outputs = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=current_attention_mask,
                        use_cache=False,
                    )

            logits = outputs.logits[:, -1, :] / temperature
            logits = logits_processors(current_input_ids, logits)
            probs = F.softmax(logits, dim=-1)

            sampled = torch.multinomial(probs, num_samples=1)
            log_prob = torch.log(torch.gather(probs, dim=-1, index=sampled) + 1e-8)

            if grad_tracking:
                log_probs.append(log_prob.clone().squeeze(1))  # [B]

            for i in range(B):
                token_id = sampled[i].item()
                sampled_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                if not finished[i]:
                    if token_id == eos_token_id:
                        finished[i] = True
                    else:
                        generated_token_ids[i].append(token_id)

                        # ```: 코드 블록 두 번째 등장 시 종료
                        if sampled_text.strip() == "```":
                            code_fence_counts[i] += 1
                            if code_fence_counts[i] >= 2:
                                finished[i] = True

            current_input_ids = torch.cat([current_input_ids, sampled.detach()], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(sampled)], dim=1)

            if hasattr(outputs, "past_key_values"):
                outputs.past_key_values = None
            del inputs_embeds, outputs, logits, probs, sampled, log_prob

            if finished.all():
                break

        if log_probs:
            final_log_probs = torch.stack(log_probs, dim=1)  # [B, T']
        else:
            final_log_probs = torch.zeros(B, 1, device=self.device)

        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        del inputs, input_ids, attention_mask, current_input_ids, current_attention_mask
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "texts": generated_texts,
            "log_probs": final_log_probs,
        }
    

    def sample_header_and_cpp_with_partial_grad(self, requirements: list[str], max_track_tokens: int = 30) -> dict:
        result = {
            "header_texts": [],
            "header_log_probs": [],
            "cpp_texts": [],
            "cpp_log_probs": []
        }

        for req in requirements:
            # --- Header ---
            header_prompt = self.create_prompt_header(req)
            header_out = self.sample_code_batch_with_partial_grad(
                [header_prompt], max_track_tokens=max_track_tokens
            )

            header_code = header_out["texts"][0]
            header_log_prob = header_out["log_probs"]
            result["header_texts"].append(header_code)
            result["header_log_probs"].append(header_log_prob)

            for k in list(header_out.keys()):
                header_out[k] = None
            del header_out
            torch.cuda.empty_cache()

            # --- CPP ---
            cpp_prompt = self.create_prompt_cpp(req, header_code)
            cpp_out = self.sample_code_batch_with_partial_grad(
                [cpp_prompt], max_track_tokens=max_track_tokens
            )

            cpp_code = cpp_out["texts"][0]
            cpp_log_prob = cpp_out["log_probs"]
            result["cpp_texts"].append(cpp_code)
            result["cpp_log_probs"].append(cpp_log_prob)

            for k in list(cpp_out.keys()):
                cpp_out[k] = None
            del cpp_out
            torch.cuda.empty_cache()

        max_header_len = max(lp.shape[1] for lp in result["header_log_probs"])
        max_cpp_len = max(lp.shape[1] for lp in result["cpp_log_probs"])

        result["header_log_probs"] = torch.cat([
            F.pad(lp, (0, max_header_len - lp.shape[1]), value=0.0)
            for lp in result["header_log_probs"]
        ], dim=0).to(self.device)

        result["cpp_log_probs"] = torch.cat([
            F.pad(lp, (0, max_cpp_len - lp.shape[1]), value=0.0)
            for lp in result["cpp_log_probs"]
        ], dim=0).to(self.device)

        return result
    

    def sample_header_with_partial_grad(self, requirements: list[str], max_track_tokens: int = 30) -> dict:
        result = {
            "header_texts": [],
            "header_log_probs": []
        }

        for req in requirements:
            header_prompt = self.create_short_prompt_header(req)
            header_out = self.sample_code_batch_with_partial_grad(
                [header_prompt], max_track_tokens=max_track_tokens
            )

            header_code = header_out["texts"][0]
            header_log_prob = header_out["log_probs"]

            result["header_texts"].append(header_code)
            result["header_log_probs"].append(header_log_prob)

            # 메모리 정리
            for k in list(header_out.keys()):
                header_out[k] = None
            del header_out
            torch.cuda.empty_cache()

        # Padding + Stacking
        max_len = max(lp.shape[1] for lp in result["header_log_probs"])
        result["header_log_probs"] = torch.cat([
            F.pad(lp, (0, max_len - lp.shape[1]), value=0.0)
            for lp in result["header_log_probs"]
        ], dim=0).to(self.device)

        return result
    

    def sample_cpp_with_partial_grad(self, requirements: list[str], header_texts: list[str], max_track_tokens: int = 30) -> dict:
        assert len(requirements) == len(header_texts), "Length mismatch between requirements and headers"

        result = {
            "cpp_texts": [],
            "cpp_log_probs": []
        }

        for req, header_code in zip(requirements, header_texts):
            cpp_prompt = self.create_short_prompt_cpp(req, header_code)
            cpp_out = self.sample_code_batch_with_partial_grad(
                [cpp_prompt], max_track_tokens=max_track_tokens
            )

            cpp_code = cpp_out["texts"][0]
            cpp_log_prob = cpp_out["log_probs"]

            result["cpp_texts"].append(cpp_code)
            result["cpp_log_probs"].append(cpp_log_prob)

            for k in list(cpp_out.keys()):
                cpp_out[k] = None
            del cpp_out
            torch.cuda.empty_cache()

        # Padding + Stacking
        max_len = max(lp.shape[1] for lp in result["cpp_log_probs"])
        result["cpp_log_probs"] = torch.cat([
            F.pad(lp, (0, max_len - lp.shape[1]), value=0.0)
            for lp in result["cpp_log_probs"]
        ], dim=0).to(self.device)

        return result



        # Padding and stacking (CPU)
        max_header_len = max(lp.shape[1] for lp in result["header_log_probs"])
        max_cpp_len = max(lp.shape[1] for lp in result["cpp_log_probs"])

        result["header_log_probs"] = torch.cat([
            F.pad(lp, (0, max_header_len - lp.shape[1]), value=0.0)
            for lp in result["header_log_probs"]
        ], dim=0).to(self.device)

        result["cpp_log_probs"] = torch.cat([
            F.pad(lp, (0, max_cpp_len - lp.shape[1]), value=0.0)
            for lp in result["cpp_log_probs"]
        ], dim=0).to(self.device)

        return result
    

    def compute_value(self, prompts: list[str], responses: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()

        full_inputs = [p + r for p, r in zip(prompts, responses)]

        inputs = self.tokenizer(
            full_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        value_total, value_disc, value_format = self.value_head(hidden_states.float(), attention_mask)

        return value_total.squeeze(-1), value_disc.squeeze(-1), value_format.squeeze(-1)




# Main
if __name__ == "__main__":
    save_path = "./checkpoint_1_3b"
    model = CodeGenerator(load_path=save_path)

    model.save("./checkpoint_1_3b")

    print("\n[Example using sample_header_and_cpp_with_partial_grad()]")
    test_requirements = [
        "Implement a character class that has health and mana attributes.",
    ]
    with torch.no_grad():
        generated_responses = model.sample_header_and_cpp_with_partial_grad(test_requirements)
    print(generated_responses["header_texts"][0])
    print("============")
    print(generated_responses["cpp_texts"][0])
    
    # print("============")
    # print(model.generate_header_and_cpp("Create a character that prints a message to the screen when the game starts."))