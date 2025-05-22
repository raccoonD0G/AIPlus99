import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import PreTrainedModel

class CodeDiscriminator(nn.Module):
    def __init__(
        self,
        load_path=None,
        model_name="Qwen/Qwen1.5-0.5B",
        max_length=1024,
        hidden_dim=512,
        use_attention_pooling=True,
        quantization="8bit",
    ):
        super().__init__()
        self.max_length = max_length
        self.use_attention_pooling = use_attention_pooling

        # === Tokenizer ===
        tokenizer_path = os.path.join(load_path, "tokenizer") if load_path and os.path.exists(os.path.join(load_path, "tokenizer")) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        # === Quant config ===
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None

        # === Load or Init Model ===
        if load_path and os.path.exists(load_path):
            print(f"Loading discriminator from {load_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quant_config,
                torch_dtype=torch.float16
            )

            base_model = prepare_model_for_kbit_training(base_model)
            self.encoder = PeftModel.from_pretrained(base_model, load_path)
        else:
            print("Initializing new discriminator...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quant_config,
                torch_dtype=torch.float16
            )

            base_model = prepare_model_for_kbit_training(base_model)

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"  # Qwen은 causal LM
            )
            self.encoder = get_peft_model(base_model, lora_config)

        # === Attention Pooling (Optional) ===
        if self.use_attention_pooling:
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=self.encoder.config.hidden_size,
                num_heads=8,
                batch_first=True
            )
            self.attention_query = nn.Parameter(
                torch.randn(1, 1, self.encoder.config.hidden_size)
            )

        # === Classifier Head ===
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.config.hidden_size),
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier.to(self.encoder.device)

        # === Load Classifier Head if available ===
        classifier_path = os.path.join(load_path or "", "classifier.pt")
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
            print(f"Loaded classifier head from {classifier_path}")
        else:
            print("No classifier head found. Initialized randomly.")

        self.encoder.gradient_checkpointing_enable()

    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)

        if isinstance(self.encoder, PeftModel):
            print("Saving LoRA adapter...")
            self.encoder.save_pretrained(save_path)
        else:
            print("Warning: encoder is not a PeftModel, skipping adapter save.")

        classifier_path = os.path.join(save_path, "classifier.pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        print(f"Saved classifier head to {classifier_path}")

        tokenizer_path = os.path.join(save_path, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def forward(self, code_list):
        if isinstance(code_list, str):
            code_list = [code_list]

        inputs = self.tokenizer(
            code_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.encoder.device)

        outputs = self.encoder(**inputs, use_cache=False, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

        if self.use_attention_pooling:
            B = hidden.size(0)
            query = self.attention_query.expand(B, -1, -1)
            attn_output, _ = self.attention_pool(query, hidden, hidden)
            pooled = attn_output.squeeze(1)
        else:
            pooled = hidden[:, 0, :]

        logits = self.classifier(pooled).squeeze(-1)
        logits = logits.clamp(min=-10, max=10)
        return logits
