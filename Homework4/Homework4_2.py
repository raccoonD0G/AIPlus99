import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from langchain_community.tools import DuckDuckGoSearchRun
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F

import json

CACHE_FILE = "search_cache.json"

# 캐시 로드
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        rag_cache = json.load(f)
else:
    rag_cache = {}

search_tool = DuckDuckGoSearchRun()

def get_cached_or_search(question: str) -> str:
    search_query = summarize_for_search_with_gemma(question)  # 요약

    if search_query in rag_cache:
        return rag_cache[search_query]

    try:
        result = search_tool.run(search_query)
    except Exception as e:
        result = f"검색 실패: {e}"

    rag_cache[search_query] = result

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(rag_cache, f, ensure_ascii=False, indent=2)

    return result

def summarize_for_search_with_gemma(question: str) -> str:
    prompt = f"""
반드시 한국어로 대답해! 너는 질문을 웹 검색에 적합하도록 간단하고 명확하게 요약해주는 요약기야.
긴 코드 블록이나 설명이 있어도, 요지를 뽑아서 간결한 자연어 검색어로 바꿔줘.
반드시 64 토큰 이내로 요약해야 해 반드시!

[질문]
{question}

[검색용 요약]
""".strip()

    inputs = gen_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = generator.generate(**inputs, max_new_tokens=64)
    summary = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[검색용 요약]" in summary:
        summary = summary.split("[검색용 요약]")[-1]

    return summary.strip()


class GemmaForClassification(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int = 2):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            attn_implementation="eager"
        )
        self.backbone = base_model.model  # Transformer만 추출


        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, num_labels)
        )


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        pooled = outputs.last_hidden_state[:, -1]  # 마지막 토큰의 hidden state
        logits = self.classifier(pooled)
        return logits


# 모델 로딩
gen_tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b-it")
generator = AutoModelForCausalLM.from_pretrained("unsloth/gemma-2-2b-it", attn_implementation="eager").to("cuda")

# Generator 일부 레이어만 학습
for param in generator.parameters():
    param.requires_grad = False

train_layer_ids = [22, 23, 24, 25]
for name, param in generator.named_parameters():
    if (
        any(name.startswith(f"model.layers.{i}") for i in train_layer_ids)
        or name.startswith("model.norm")
    ):
        param.requires_grad = True
    else:
        param.requires_grad = False

print("학습되는 Gemma:")
for name, param in generator.named_parameters():
    if param.requires_grad:
        print(" -", name)

disc_tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b-it")
discriminator = GemmaForClassification("unsloth/gemma-2-2b-it", num_labels=2).to("cuda")

# Discriminator는 classifier만 학습
for name, param in discriminator.named_parameters():
    param.requires_grad = "classifier" in name

print("학습되는 Discriminator:")
for name, param in discriminator.named_parameters():
    if param.requires_grad:
        print(f" - {name}")

# DuckDuckGo 검색기 초기화
search_tool = DuckDuckGoSearchRun()
rag_cache = {}  # RAG 캐시 추가

# 옵티마이저와 손실함수
gen_optimizer = AdamW(filter(lambda p: p.requires_grad, generator.parameters()), lr=5e-5)
disc_optimizer = AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=5e-5)
loss_fn = CrossEntropyLoss()
num_epochs = 10

# Gemma RAG 기반 응답 생성 함수
def generate_with_rag_gemma(question: str) -> str:
    context = get_cached_or_search(question)
    prompt = f"""
반드시 한국어로 대답해! 너는 코드 리뷰 전문가야. 아래 검색 결과와 코드 조각을 참고해서 코드에 어떤 문제가 있고 어떻게 개선할 수 있는지 조언을 줘.
추측하지 말고, 검색 결과와 문맥 안의 정보만 사용해. 실제 코드 리뷰처럼 구조적인 피드백을 줘.

[검색 결과]
{context}

[질문]
{question}

[답변]
""".strip()

    inputs = gen_tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = generator.generate(**inputs, max_new_tokens=4096)
    decoded = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1]

    answer = decoded.strip().lstrip("\\n").strip('"').strip("'")
    rag_cache[question] = answer
    return answer

# 훈련 함수
def train_gan(generator, discriminator, train_loader, test_loader):
    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} 시작\n" + "="*40)

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] 훈련 중"):
            questions = batch["questions"]
            real_answers = batch["real_answers"]

            # === Generator 업데이트 ===
            generated_answers = [generate_with_rag_gemma(q) for q in questions]

            fake_inputs = disc_tokenizer(questions, generated_answers, return_tensors="pt", padding=True, truncation=True).to("cuda")
            fake_output = discriminator(**fake_inputs)
            fake_labels = torch.ones_like(fake_output[:, 0], dtype=torch.long)
            loss_G = loss_fn(fake_output, fake_labels)

            generator.zero_grad()
            loss_G.backward()
            gen_optimizer.step()

            # === Discriminator 업데이트 ===
            real_inputs = disc_tokenizer(questions, real_answers, return_tensors="pt", padding=True, truncation=True).to("cuda")
            real_output = discriminator(**real_inputs)
            real_labels = torch.zeros_like(real_output[:, 0], dtype=torch.long)
            loss_real = loss_fn(real_output, real_labels)

            fake_inputs = disc_tokenizer(questions, generated_answers, return_tensors="pt", padding=True, truncation=True).to("cuda")
            fake_output = discriminator(**fake_inputs)
            loss_fake = loss_fn(fake_output, fake_labels)

            loss_D = loss_real + loss_fake
            discriminator.zero_grad()
            loss_D.backward()
            disc_optimizer.step()

        # 저장
        save_dir = "./checkpoints/gemma_gpt_gan"

        # Generator 저장
        generator.save_pretrained(save_dir)
        gen_tokenizer.save_pretrained(save_dir)
        # Discriminator 저장
        torch.save(discriminator.state_dict(), f"{save_dir}/discriminator.pt")


        print(f"\nEpoch {epoch + 1} 완료")
        print(f"   Generator Loss: {loss_G.item():.4f} | Discriminator Loss: {loss_D.item():.4f}")

        # 테스트 출력
        print(f"\nEpoch 완료 - Gemma, GPT-4o 응답 비교\n")
        sampled = random.sample(list(test_loader.dataset), k=3)

        for i, example in enumerate(sampled):
            test_question = example["question"]
            real_answer = example["real_answer"]

            gemma_answer = generate_with_rag_gemma(test_question)
            
            print(f"\n질문 {i+1}:\n{test_question}")
            print(f"\n정답:\n{real_answer}")
            print(f"\nGemma 응답:\n{gemma_answer}")
            print("=" * 100)

# 데이터 로드
dataset = load_dataset("json", data_files="generated_data.json", split="train")
dataset = dataset.shuffle(seed=42).select(range(100))
split = dataset.train_test_split(test_size=0.2)
train_data = split["train"]
test_data = split["test"]

from torch.utils.data import DataLoader

# 배치 크기 설정
BATCH_SIZE = 8

# 질문/답변만 추출
def collate_fn(batch):
    questions = [example["question"] for example in batch]
    real_answers = [example["real_answer"] for example in batch]
    return {"questions": questions, "real_answers": real_answers}

# DataLoader 생성
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

# 훈련 시작
train_gan(generator, discriminator, train_loader, test_loader)
