# PPO-based training loop for CodeGenerator
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn
from copy import deepcopy
from tqdm import tqdm
from code_generator_1_3b import CodeGenerator
from code_discriminator import CodeDiscriminator
from requirement_to_code_dataset import get_train_dataloader
from value_head import ValueHead
from bitsandbytes.optim import AdamW8bit
import gc

def batched_discriminator(discriminator, texts, batch_size=1):
    preds = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        logits = discriminator(chunk)
        preds.append(logits.cpu())
    return torch.cat(preds, dim=0).to(discriminator.classifier[0].weight.device)

def safe_del(*vars_to_del):
    for var in vars_to_del:
        if var in locals() or var in globals():
            try:
                del globals()[var]
            except KeyError:
                try:
                    del locals()[var]
                except:
                    pass

def ppo_train_separated(generator, discriminator, tokenizer, train_loader, epochs=3, lr=2e-5, clip_eps=0.2, value_coef=0.5, save_path="./checkpoint_1_3b"):
    generator.train()
    discriminator.train()

    ref_generator = CodeGenerator(load_path=save_path)
    ref_generator.eval()
    for param in ref_generator.parameters():
        param.requires_grad = False

    g_optimizer = AdamW8bit(generator.parameters(), lr=lr)
    d_optimizer = AdamW8bit(discriminator.parameters(), lr=1e-4)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        loop = tqdm(train_loader)

        for step, batch in enumerate(loop):
            torch.cuda.empty_cache()

            requirements = batch["requirement"]
            reference_headers = batch["header_code"]
            reference_cpps = batch["cpp_code"]
            prompts = [generator.create_prompt_header(r) for r in requirements]

            # === [Phase 1] Header PPO 학습 ===
            header_output = generator.sample_header_with_partial_grad(requirements)
            print("\n===== .h =====")
            print(header_output["header_texts"][0])
            with torch.no_grad():
                ref_header_output = ref_generator.sample_header_with_partial_grad(requirements)

            # Rewards
            with torch.no_grad():
                header_outputs = header_output["header_texts"]
                d_scores = torch.sigmoid(discriminator(header_outputs)).squeeze()
                header_rewards = 2 * (d_scores - 0.5)

                format_penalties = torch.tensor([
                    sum([
                        -1.0 if not text.strip().startswith("```cpp") else 0.0,
                        -2.0 if not text.strip().endswith("```") else 0.0,
                        -1.0 if "#pragma once" not in text else 0.0,
                        -1.0 if ("UCLASS" not in text and "USTRUCT" not in text) else 0.0,
                    ])
                    for text in header_outputs
                ], dtype=torch.float32, device=header_rewards.device)

                header_rewards = header_rewards + format_penalties

            # Log-probs
            log_probs = header_output["header_log_probs"].mean(dim=1)
            ref_log_probs = ref_header_output["header_log_probs"].mean(dim=1)
            log_ratio = (log_probs - ref_log_probs).clamp(min=-10, max=10)
            ratios = torch.exp(log_ratio)

            # Values & Advantages
            values_total, values_disc, values_format = generator.compute_value(prompts, header_output["header_texts"])

            advantages = header_rewards - values_total.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # PPO Loss
            ppo_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            ).mean()

            # Value Loss
            reward_disc = d_scores
            reward_format = format_penalties

            loss_disc = nn.functional.mse_loss(values_disc, reward_disc.to(values_disc.dtype))
            loss_format = nn.functional.mse_loss(values_format, reward_format.to(values_format.dtype))
            value_loss = loss_disc + loss_format

            # SFT loss
            joined_inputs = [p + r for p, r in zip(prompts, reference_headers)]
            tokenized = tokenizer(joined_inputs, return_tensors="pt", padding=True, truncation=True, max_length=generator.max_length).to(generator.device)
            labels = tokenized["input_ids"].clone()

            prompt_tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=generator.max_length)
            prompt_lens = [len(p[p != tokenizer.pad_token_id]) for p in prompt_tokenized["input_ids"]]
            for i, prompt_len in enumerate(prompt_lens):
                labels[i, :prompt_len] = -100

            sft_outputs = generator(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], labels=labels)
            sft_loss = sft_outputs.loss

            # Total loss
            sft_coef = 0.05
            total_loss = ppo_loss + value_coef * value_loss + sft_coef * sft_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_optimizer.step()
            g_optimizer.zero_grad()

            # === Train D on h ===
            if header_rewards.mean().item() > -0.4 or (epoch == 0 and step == 0):
                print("\n[Step {}] Training header discriminator...".format(step))
                all_codes = reference_headers + header_output["header_texts"]
                all_labels = torch.tensor([1]*len(reference_headers) + [0]*len(header_output["header_texts"]), dtype=torch.float32).to(generator.device)

                d_preds = batched_discriminator(discriminator, all_codes)
                d_loss = nn.BCEWithLogitsLoss()(d_preds, all_labels)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()


            loop.set_description(f"[Epoch {epoch+1}] Header")
            loop.set_postfix({
                "total_loss": total_loss.item(),
                "ppo": ppo_loss.item(),
                "value": value_loss.item(),
                "sft": sft_loss.item(),
                "reward": header_rewards.mean().item()
            })

            # === Detach all tensors ===
            for t in [
                "total_loss", "ppo_loss", "value_loss", "sft_loss",
                "values", "values_total", "values_disc", "values_format",
                "header_rewards", "log_probs", "ref_log_probs", "advantages",
                "reward_disc", "reward_format", "loss_disc", "loss_format"
            ]:
                if t in locals() and isinstance(locals()[t], torch.Tensor):
                    locals()[t] = locals()[t].detach()

            # === Bulk delete ===
            safe_del(
                "total_loss", "ppo_loss", "value_loss", "sft_loss",
                "values", "values_total", "values_disc", "values_format",
                "header_rewards", "log_probs", "ref_log_probs", "advantages",
                "reward_disc", "reward_format", "loss_disc", "loss_format",

                "tokenized", "labels", "prompt_tokenized", "prompt_lens",
                "sft_outputs",

                "sample_output", "ref_output",
                "header_output", "ref_header_output", "cpp_output", "ref_cpp_output",

                "generated_codes", "reference_codes",
                "joined_inputs", "cpp_prompts", "cpp_concat"
            )

            # === Garbage collection ===
            gc.collect()
            torch.cuda.empty_cache()


            # === [Phase 2] CPP PPO 학습 ===
            cpp_output = generator.sample_cpp_with_partial_grad(requirements, reference_headers)
            print("\n===== .cpp =====")
            print(cpp_output["cpp_texts"][0])
            with torch.no_grad():
                ref_cpp_output = ref_generator.sample_cpp_with_partial_grad(requirements, reference_headers)
            cpp_prompts = [generator.create_prompt_cpp(r, h) for r, h in zip(requirements, reference_headers)]

            with torch.no_grad():
                cpp_outputs = cpp_output["cpp_texts"]
                d_scores = torch.sigmoid(discriminator(cpp_outputs)).squeeze()
                cpp_rewards = 2 * (d_scores - 0.5)

                format_penalties = torch.tensor([
                    sum([
                        -1.0 if not text.strip().startswith("```cpp") else 0.0,
                        -2.0 if not text.strip().endswith("```") else 0.0,
                        -1.0 if "::" not in text else 0.0,
                    ])
                    for text in cpp_outputs
                ], dtype=torch.float32, device=cpp_rewards.device)

                cpp_rewards = cpp_rewards + format_penalties

            # Log-prob ratio
            log_probs = cpp_output["cpp_log_probs"].mean(dim=1)
            ref_log_probs = ref_cpp_output["cpp_log_probs"].mean(dim=1)
            log_ratio = (log_probs - ref_log_probs).clamp(min=-10, max=10)
            ratios = torch.exp(log_ratio)

            # Values & Advantages
            values_total, values_disc, values_format = generator.compute_value(cpp_prompts, cpp_output["cpp_texts"])

            advantages = cpp_rewards - values_total.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # PPO Loss
            ppo_loss = -torch.min(
                ratios * advantages,
                torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            ).mean()

            # Value Loss
            reward_disc = d_scores
            reward_format = format_penalties

            loss_disc = nn.functional.mse_loss(values_disc, reward_disc.to(values_disc.dtype))
            loss_format = nn.functional.mse_loss(values_format, reward_format.to(values_format.dtype))
            value_loss = loss_disc + loss_format

            # SFT Loss
            joined_inputs = [p + r for p, r in zip(cpp_prompts, reference_cpps)]
            tokenized = tokenizer(joined_inputs, return_tensors="pt", padding=True, truncation=True, max_length=generator.max_length).to(generator.device)
            labels = tokenized["input_ids"].clone()

            prompt_tokenized = tokenizer(cpp_prompts, return_tensors="pt", padding=True, truncation=True, max_length=generator.max_length)
            prompt_lens = [len(p[p != tokenizer.pad_token_id]) for p in prompt_tokenized["input_ids"]]
            for i, prompt_len in enumerate(prompt_lens):
                labels[i, :prompt_len] = -100

            sft_outputs = generator(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], labels=labels)
            sft_loss = sft_outputs.loss

            total_loss = ppo_loss + value_coef * value_loss + sft_coef * sft_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            g_optimizer.step()
            g_optimizer.zero_grad()

            # === Train D on cpp ===
            if cpp_rewards.mean().item() > -0.4 or (epoch == 0 and step == 0):
                print("\n[Step {}] Training cpp discriminator...".format(step))
                all_codes = reference_cpps + cpp_output["cpp_texts"]
                all_labels = torch.tensor([1]*len(reference_cpps) + [0]*len(cpp_output["cpp_texts"]), dtype=torch.float32).to(generator.device)

                d_preds = batched_discriminator(discriminator, all_codes)
                d_loss = nn.BCEWithLogitsLoss()(d_preds, all_labels)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()


            loop.set_description(f"[Epoch {epoch+1}] CPP")
            loop.set_postfix({
                "total_loss": total_loss.item(),
                "ppo": ppo_loss.item(),
                "value": value_loss.item(),
                "sft": sft_loss.item(),
                "reward": cpp_rewards.mean().item()
            })

            # Clean-up
            # === Detach all tensors ===
            for t in [
                "total_loss", "ppo_loss", "value_loss", "sft_loss",
                "values", "values_total", "values_disc", "values_format",
                "cpp_rewards", "log_probs", "ref_log_probs", "advantages"
            ]:
                if t in locals() and isinstance(locals()[t], torch.Tensor):
                    locals()[t] = locals()[t].detach()

            # === Bulk delete ===
            safe_del(
                # Losses & core tensors
                "total_loss", "ppo_loss", "value_loss", "sft_loss",
                "values", "values_total", "values_disc", "values_format",
                "log_probs", "ref_log_probs", "advantages",
                "cpp_rewards", "reward_disc", "reward_format",
                "loss_disc", "loss_format",

                # Tokenizer outputs
                "tokenized", "labels", "prompt_tokenized", "prompt_lens",

                # Model outputs
                "sft_outputs",

                # Sample/reference
                "sample_output", "ref_output", "header_output", "ref_header_output",
                "cpp_output", "ref_cpp_output",

                # Misc
                "generated_codes", "reference_codes",
                "joined_inputs", "cpp_prompts", "cpp_concat"
            )

            # === Final cleanup ===
            gc.collect()
            torch.cuda.empty_cache()

            # === 주기적 ref_generator 동기화 ===
            if step % 10 == 0:
                print(f"\n[Step {step}] Ref generator updated.")
                del ref_generator
                ref_generator = deepcopy(generator)
                ref_generator.eval()
                for p in ref_generator.parameters():
                    p.requires_grad = False

        # Save checkpoint
        generator.save(save_path)
        torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator.pt"))

        with torch.no_grad():
            generated_responses = generator.sample_header_and_cpp_with_partial_grad(["Create a character class with health and mana properties"])
        print("\n===== .h =====")
        print(generated_responses["header_texts"][0])
        print("\n===== .cpp =====")
        print(generated_responses["cpp_texts"][0])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./checkpoint_1_3b"

    generator = CodeGenerator(load_path=save_path)
    tokenizer = generator.tokenizer

    discriminator = CodeDiscriminator().to(device)
    d_path = os.path.join(save_path, "discriminator.pt")
    if os.path.exists(d_path):
        discriminator.load_state_dict(torch.load(d_path, map_location=device))
        print("Discriminator loaded.")
    else:
        print("Training discriminator from scratch.")

    with torch.no_grad():
            generated_responses = generator.sample_header_and_cpp_with_partial_grad(["Create a character class with health and mana properties"])
    print("\n===== .h =====")
    print(generated_responses["header_texts"][0])
    print("\n===== .cpp =====")
    print(generated_responses["cpp_texts"][0])

    train_loader = get_train_dataloader("unreal_code_dataset.jsonl", batch_size=2, shuffle=True, limit= 90)
    
    ppo_train_separated(generator, discriminator, tokenizer, train_loader, epochs=3, save_path=save_path)

