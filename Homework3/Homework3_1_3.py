import nltk
nltk.download('punkt')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from openai import OpenAI
# from google.colab import userdata
import re
import fitz

import fitz
import re

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    true = labels.argmax(axis=-1)
    acc = accuracy_score(true, preds)
    return {"accuracy": acc}

def extract_questions_from_pdf(pdf_path, max_questions=34):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])

    range_tags = re.findall(r'\[(\d+)[~～](\d+)\]', full_text)
    paragraph_ranges = [(int(start), int(end)) for start, end in range_tags]

    raw_blocks = re.split(r'\n?\s*(\d{1,2})[.)번]?\s+', full_text)

    questions = []
    current_shared_paragraph = ""
    current_range = None
    prev_text = raw_blocks[0]

    for i in range(1, len(raw_blocks) - 1, 2):
        number = int(raw_blocks[i])
        if number > max_questions:
            break  # 34번까지만

        body = raw_blocks[i + 1]
        in_shared_range = False
        for r in paragraph_ranges:
            if r[0] <= number <= r[1]:
                in_shared_range = True
                if current_range != r:
                    pattern = re.compile(rf'\[{r[0]}[~～]{r[1]}\](.*?)(?=\n\s*{r[0]}[.)번])', re.DOTALL)
                    match = pattern.search(full_text)
                    current_shared_paragraph = match.group(1).strip() if match else ""
                    current_range = r
                break

        choices_match = re.findall(
            r'①(.*?)②(.*?)③(.*?)④(.*?)⑤(.*?)(?=\n\d{1,2}[.)번]|\n\n|\Z)',
            body,
            re.DOTALL
        )
        if choices_match:
            choices = choices_match[0]
            question_match = re.split(r'①', body, maxsplit=1)
            question_text = question_match[0].strip()
            paragraph = current_shared_paragraph if in_shared_range else prev_text.strip()

            questions.append({
                "number": number,
                "paragraph": paragraph,
                "question": question_text,
                "choices": [o.strip() for o in choices]
            })

            prev_text = ""
        else:
            prev_text += "\n" + body

    return questions


def extract_common_subject_answers(pdf_path, max_questions=34):
    doc = fitz.open(pdf_path)
    text = doc[0].get_text()

    symbol_to_index = {'①': 0, '②': 1, '③': 2, '④': 3, '⑤': 4}
    matches = re.findall(r'(\d+)\s*([①-⑤])\s*\d+', text)

    answers = {}
    for number, symbol in matches:
        num = int(number)
        if 1 <= num <= max_questions:
            answers[num] = symbol_to_index[symbol]

    correct_indices = [answers[i] for i in range(1, max_questions + 1) if i in answers]
    
    if len(correct_indices) < max_questions:
        print(f"경고: 추출된 정답 수 {len(correct_indices)}개가 기대값 {max_questions}보다 적습니다.")

    return correct_indices

questions = extract_questions_from_pdf("Exam2.pdf")
answers = extract_common_subject_answers("Answer2.pdf")

import json

def print_questions(questions, max_items=None):
    if max_items is not None:
        questions = questions[:max_items]

    print(json.dumps(questions, ensure_ascii=False, indent=2))


# 정답 붙이기
for i in range(min(len(questions), len(answers))):
    questions[i]["answer"] = answers[i]

print (answers);
# 출력
print_questions(questions, max_items=34)


import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModel

class ExamCrossAttention(nn.Module):
    def __init__(self, num_labels=5, hidden_size=768, num_heads=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained("klue/bert-base")
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        for name, param in self.bert.named_parameters():
            if "encoder.layer.10." in name or "encoder.layer.11." in name:
                param.requires_grad = True
                
        print("Trainable BERT Layers:")
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                print(f"{name}")
    
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(3)  # CrossAttention 3층
        ])


        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, num_labels)
        )

        self.last_attn_weights = None
        
    def forward(self, input_ids_question, attention_mask_question, input_ids_paragraph, attention_mask_paragraph, labels=None):
        # 1. 질문 인코딩
        q_output = self.bert(input_ids=input_ids_question, attention_mask=attention_mask_question)
        question_embed = q_output.last_hidden_state  # (B, Lq, D)

        # 2. 지문 인코딩
        c_output = self.bert(input_ids=input_ids_paragraph, attention_mask=attention_mask_paragraph)
        paragraph_embed = c_output.last_hidden_state  # (B, Lc, D)

        # 3. Cross-Attention
        self.last_attn_weights = []
        x = question_embed
        for i, layer in enumerate(self.cross_layers):
            attn_output, attn_weights = layer[0](  # layer[0]은 MultiheadAttention
                query=x,
                key=paragraph_embed,
                value=paragraph_embed,
                key_padding_mask=(attention_mask_paragraph == 0),
                average_attn_weights=False
            )
            x = layer[1](x + attn_output)  # layer[1]은 LayerNorm
            self.last_attn_weights.append(attn_weights.detach())
        attended = x

        # 5. 질문 토큰에 대한 pooling
        mean_pooled = attended.mean(dim=1)       # (B, D)
        max_pooled = attended.max(dim=1).values  # (B, D)

        pooled = (mean_pooled + max_pooled) / 2  # (B, D)


        logits = self.classifier(pooled)

        if labels is not None:
            labels_idx = labels.argmax(dim=1)
            loss = nn.CrossEntropyLoss()(logits, labels_idx)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
    


from torch.utils.data import Dataset

class MCQDataset(Dataset):
    def __init__(self, questions, tokenizer, max_q_len=512, max_p_len=512):
        self.questions = [q for q in questions if "answer" in q]
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        question_text = q["question"]
        paragraph_text = q["paragraph"]  # 지문도 함께 사용
        label_idx = q["answer"]

        # 질문 인코딩
        q_encoding = self.tokenizer(
            question_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_q_len,
            return_tensors="pt"
        )

        # 지문 인코딩
        p_encoding = self.tokenizer(
            paragraph_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_p_len,
            return_tensors="pt"
        )

        # 레이블 (one-hot)
        one_hot = torch.zeros(5)
        one_hot[label_idx] = 1

        return {
            "input_ids_question": q_encoding["input_ids"].squeeze(0),
            "attention_mask_question": q_encoding["attention_mask"].squeeze(0),
            "input_ids_paragraph": p_encoding["input_ids"].squeeze(0),
            "attention_mask_paragraph": p_encoding["attention_mask"].squeeze(0),
            "labels": one_hot
        }


from transformers import TrainingArguments, Trainer

train_questions = []
test_questions = []

for i in range(1, 3):
    exam_file = f"Exam{i}.pdf"
    answer_file = f"Answer{i}.pdf"

    # 문제 및 정답 추출
    pdf_questions = extract_questions_from_pdf(exam_file)
    pdf_answers = extract_common_subject_answers(answer_file)

    for j in range(min(len(pdf_questions), len(pdf_answers))):
        pdf_questions[j]["answer"] = pdf_answers[j]

    train_questions.extend(pdf_questions)
    test_questions.extend(pdf_questions)
        


print(f"훈련용 문제 수: {len(train_questions)}")
print(f"테스트용 문제 수: {len(test_questions)}")

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
tokenizer.model_max_length = 512

for i in range(1, 3):
    exam_file = f"Exam{i}.pdf"
    answer_file = f"Answer{i}.pdf"

    questions = extract_questions_from_pdf(exam_file)
    answers = extract_common_subject_answers(answer_file)

    print(f"[Exam{i}] 문제 수: {len(questions)}, 정답 수: {len(answers)}")
    
# 학습용 Dataset 생성
train_dataset = MCQDataset(train_questions, tokenizer)

# 테스트용 Dataset 생성
test_dataset = MCQDataset(test_questions, tokenizer)

model = ExamCrossAttention()


training_args = TrainingArguments(
    output_dir="./Exam",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=60,
    eval_strategy="epoch",     
    save_strategy="epoch",           
    logging_strategy="epoch",
    load_best_model_at_end=True,     
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=1
)


from transformers import TrainerCallback
from tqdm import tqdm

class LogAccuracyCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        torch.save(model.state_dict(), "trained_model.pt")


trainer  = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[LogAccuracyCallback()]
)

import matplotlib.pyplot as plt

def plot_accuracy_from_trainer(trainer):
    log_history = trainer.state.log_history
    eval_steps = []
    eval_accs = []

    for entry in log_history:
        if "eval_accuracy" in entry:
            step = entry.get("epoch", entry.get("step", None))
            eval_steps.append(step)
            eval_accs.append(entry["eval_accuracy"])

    if not eval_accs:
        print("정확도 기록이 없습니다.")
        return

    plt.plot(eval_steps, eval_accs, marker='o', label="Eval Accuracy")
    plt.title("Evaluation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# trainer.train()

model.load_state_dict(torch.load("trained_model.pt"))

from openai import OpenAI

client = OpenAI(api_key="")

def emphasize_paragraph_with_gpt(paragraph, question, model, tokenizer, train_questions, device="cuda", top_k=20):
    """
    - 모델이 생성한 강조 예시 3개를 넣고,
    - GPT에게 정식 강조 문단 생성을 요청함
    """
    # 1. 모델 기반 예시 1개 생성
    few_shot_examples = []
    for i in range(1):
        ex = train_questions[i]
        q_text = ex["question"]
        if "question_plus" in ex:
            q_text += "\n\n" + ex["question_plus"]
        p_text = ex["paragraph"]

        highlighted, _ = highlight_words_with_cross_attention(
            question_text=q_text,
            paragraph_text=p_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=top_k
        )

        formatted = f"""입력 :

[지문]
{p_text}

[질문]
{q_text}

출력 :
{highlighted}"""
        few_shot_examples.append(formatted)

    example_block = "\n\n---\n\n".join(few_shot_examples)

    # 🔹 2. 최종 프롬프트 구성
    prompt = f"""
다음은 수능형 객관식 문제의 [지문]을 강조하는 작업입니다.
[지문]의 중요한 단어만 **강조**해주세요. [지문]의 단어 생략 없이 그대로 출력하되, 핵심 단어만 **강조** 표시하세요.
[질문]이 아닌 [지문]을 출력하는 겁니다. [질문]에 강조표시 해서 출력하지 마세요.

다음은 예시입니다:

{example_block}

---

이제 정식 입력입니다.

입력 :

[지문]
{paragraph}

[질문]
{question}

출력 :
"""

    # 3. GPT 호출
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
    )
    # print(f"[강조 과정] {prompt}")
    return response.choices[0].message.content.strip()


url = "https://raw.githubusercontent.com/NomaDamas/KICE_slayer_AI_Korean/master/data/2023_11_KICE.json"
df = pd.read_json(url)

# 미리보기
df.head()

def count_tokens(text):
    return len(tokenizer.encode(text))

import torch
from transformers import DistilBertModel, DistilBertTokenizer

import string

def highlight_words_with_cross_attention(question_text, paragraph_text, model, tokenizer, max_q_len=512, max_p_len=512, top_k=20, device="cuda"):

    q_enc = tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_q_len)
    p_enc = tokenizer(paragraph_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_p_len)

    input_ids_q = q_enc["input_ids"].to(device)
    mask_q = q_enc["attention_mask"].to(device)
    input_ids_p = p_enc["input_ids"].to(device)
    mask_p = p_enc["attention_mask"].to(device)

    with torch.no_grad():
        model.eval()
        _ = model(
            input_ids_question=input_ids_q,
            attention_mask_question=mask_q,
            input_ids_paragraph=input_ids_p,
            attention_mask_paragraph=mask_p
        )
        
    layer_avg = [layer.mean(dim=1)[0] for layer in model.last_attn_weights]  # (Lq, Lp) per layer
    avg_attn = torch.stack(layer_avg).mean(dim=0)  # 평균 (Lq, Lp)
    token_scores = avg_attn.mean(dim=0)  # (Lp,)

    tokens = tokenizer.convert_ids_to_tokens(input_ids_p[0], skip_special_tokens=True)

    words = []
    scores = []
    current_word = ""
    current_score_total = 0
    current_token_count = 0

    for token, score in zip(tokens, token_scores):
        if token.startswith("##"):
            token = token[2:]
            current_word += token
        else:
            if current_word:
                words.append(current_word)
                avg_score = current_score_total / max(current_token_count, 1)
                scores.append(avg_score)
            current_word = token
            current_score_total = 0
            current_token_count = 0
        current_score_total += score.item()
        current_token_count += 1

    if current_word:
        words.append(current_word)
        scores.append(current_score_total / max(current_token_count, 1))

    # top-k 단어 강조
    top_k = min(top_k, len(scores))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    highlighted_words = []
    for i, word in enumerate(words):
        # 문장 부호인지 확인
        if word in string.punctuation or word in ["``", "''", "”", "“", "’", "‘", "(", ")", ",", ".", "!", "?", "·", "…", ":", ";"]:
            # 문장 부호는 바로 앞 단어에 붙이기
            if highlighted_words:
                highlighted_words[-1] += word
            else:
                highlighted_words.append(word)
        else:
            if i in top_indices:
                highlighted_words.append(f"**{word}**")
            else:
                highlighted_words.append(word)

    highlighted_paragraph = " ".join(highlighted_words)
    return highlighted_paragraph, scores

from nltk.tokenize import sent_tokenize


def solve_question_with_emphasized_paragraph(highlighted_paragraph, question, choices, few_shot_question):
    """
    GPT가 문제를 풀기 전에 emphasize_paragraph_with_gpt()로 생성된 예시를 하나 보고 학습하도록 구성
    """
    # 1. Few-shot 예시 준비
    ex_q_text = few_shot_question["question"]
    if "question_plus" in few_shot_question:
        ex_q_text += "\n\n" + few_shot_question["question_plus"]
    ex_paragraph = few_shot_question["paragraph"]
    ex_choices = few_shot_question["choices"]
    ex_answer = few_shot_question["answer"] + 1  # 1~5 포맷

    # 강조 문단은 GPT로 생성
    try:
        ex_highlighted = emphasize_paragraph_with_gpt(
            paragraph=ex_paragraph,
            question=ex_q_text,
            model=model,
            tokenizer=tokenizer,
            train_questions=train_questions,
            device="cuda"
        )
    except Exception as e:
        print(f"[Few-shot 강조 실패] {e}")
        ex_highlighted = ex_paragraph  # fallback

    ex_choice_block = "\n".join([f"{i+1}) {c}" for i, c in enumerate(ex_choices)])

    few_shot_prompt = f"""[강조 문단]
{ex_highlighted}

[질문]
{ex_q_text}

[선택지]
{ex_choice_block}

[정답]
{ex_answer}"""

    # 2. 실제 문제
    choice_block = "\n".join([f"{i+1}) {c}" for i, c in enumerate(choices)])

    prompt = f"""
다음은 한국 수능형 객관식 문제입니다.

아래 [지문]과 [질문]을 읽고, [질문]에 가장 잘 부합하는 [선택지] 중 하나를 골라주세요.
**강조**가 적용된 [지문]과 [질문]입니다. 먼저 예시를 하나 보고, 이후 새로운 문제를 풀어주세요.

[정답]은 1, 2, 3, 4, 5 중 하나입니다.
[정답]을 숫자 하나로만 출력하세요.
그 외 텍스트를 포함하지 마세요.

예시:
{few_shot_prompt}

---

이제 새로운 문제입니다:

[강조 문단]
{highlighted_paragraph}

[질문]
{question}

[선택지]
{choice_block}

[정답]
정답 번호 하나만 출력해주세요. 예: 2
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=5,
    )
    print(f"[full prompt] {prompt}")
    result = response.choices[0].message.content.strip()
    match = re.search(r"\b([1-5])\b", result)
    return int(match.group(1)) if match else -1

import random

# 전체 평가 루프
from tqdm import tqdm

def evaluate_dataframe_with_few_shot(df, train_questions):
    correct = 0
    total_score = 0
    total_questions = 0

    train_questions = [q for q in train_questions if "answer" in q]

    progress_bar = tqdm(df.iterrows(), total=len(df), desc="Few-shot GPT 채점 중")

    for index, row in progress_bar:
        paragraph = row["paragraph"]
        problems = row["problems"]

        for i, problem in enumerate(problems):
            problem["paragraph"] = paragraph
            q_text = problem["question"]
            if "question_plus" in problem:
                q_text += "\n\n" + problem["question_plus"]

            few_shots = random.sample(train_questions, 3)
            if len(few_shots) < 3:
                continue

            correct_answer = problem["answer"]
            score = problem.get("score", 1)

            try:
                highlighted_paragraph = emphasize_paragraph_with_gpt(
                    paragraph=problem["paragraph"],
                    question=q_text,
                    model=model,
                    tokenizer=tokenizer,
                    train_questions=train_questions,
                    device="cuda"
                )
            except Exception as e:
                print(f"[강조 실패] {e}")
                continue

            few_shot_sample = random.choice(train_questions)

            gpt_answer = solve_question_with_emphasized_paragraph(
                highlighted_paragraph=highlighted_paragraph,
                question=q_text,
                choices=problem['choices'],
                few_shot_question=few_shot_sample
            )

            if gpt_answer == correct_answer:
                correct += 1
                total_score += score
            total_questions += 1

            # 현재 정답률 출력
            if total_questions > 0:
                acc = correct / total_questions * 100
                progress_bar.set_postfix(acc=f"{acc:.2f}%", correct=correct, total=total_questions)

    final_acc = correct / total_questions * 100 if total_questions > 0 else 0
    return final_acc, total_score, total_questions

# 채점 실행
accuracy, total_score, total_questions = evaluate_dataframe_with_few_shot(df, train_questions)

print(f"\n총 문제 수: {total_questions}")
print(f"정답 수: {int(accuracy * total_questions / 100)}")
print(f"정답률: {accuracy:.2f}%")
print(f"총 점수: {total_score}점")

