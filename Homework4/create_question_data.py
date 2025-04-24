import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import csv
import random
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun

import time
from tqdm import tqdm

question_prompts = [
    "초보자가 작성한 비효율적인 파이썬 코드를 보여줘. 코드만 출력해줘. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "버그가 섞여 있는 복잡한 파이썬 코드 한 조각을 만들어줘. 설명은 생략. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "파이썬에서 흔히 생기는 실수를 담은 코드 예제를 만들어줘. 코드만 보여줘. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "의도는 좋지만 비효율적인 파이썬 코드를 출력해줘. 설명은 하지 마. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "성능 문제를 유발할 수 있는 Python 코드 샘플을 생성해줘. 코드만 출력. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "아주 잘 짜여진 파이썬 코드를 보여줘. 코드만 출력. 코드 리뷰용으로 쓸거니까 주석은 달지 마."
    "가독성이 뛰어난 파이썬 코드를 보여줘. 코드만 출력. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "파이썬의 모듈화가 잘 된 예제를 보여줘. 함수 분리가 잘 된 코드를 코드만 출력해줘. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "에러 처리가 잘 되어 있는 깔끔한 파이썬 코드를 코드만 출력해줘. 주석은 넣지 마. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
    "시간 복잡도와 공간 효율성이 고려된 최적화된 파이썬 코드 예제를 보여줘. 코드만 출력해줘. 코드 리뷰용으로 쓸거니까 주석은 달지 마.",
]

# 질문 생성 체인 생성 함수
def random_question_chain():
    prompt_text = random.choice(question_prompts)
    prompt = PromptTemplate.from_template(prompt_text)
    return prompt | llm | parser

# API 키 설정
os.environ["OPENAI_API_KEY"] = ""

# GPT-4o LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.9, max_tokens=512)
parser = StrOutputParser()

# DuckDuckGo 검색 → 문서 추출
search_tool = DuckDuckGoSearchRun()

def duckduckgo_retriever(question: str) -> list[str]:
    # 코드 키워드 추출 (예: "파이썬", "버그", "함수", "리스트 컴프리헨션")
    keywords = question.split()[:4]
    query = " ".join(keywords) + " Python code review"
    result = search_tool.run(query)
    return [result]

retriever = RunnableLambda(duckduckgo_retriever)

# 질문 답변 프롬프트 (RAG 스타일)
answer_prompt = PromptTemplate.from_template("""
너는 코드 리뷰 전문가야. 아래 검색 결과와 코드 조각을 참고해서 코드에 어떤 문제가 있고 어떻게 개선할 수 있는지 조언을 줘.
추측하지 말고, 검색 결과와 문맥 안의 정보만 사용해. 실제 코드 리뷰처럼 구체적이고 구조적인 피드백을 줘.

[검색 결과]
{context}

[질문]
{question}
""")


# RAG 체인 (검색 기반 질문 답변)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | answer_prompt
    | llm
    | parser
)

# 질문 생성 함수
def generate_questions_from_gpt(num_questions=1000):
    questions = []
    seen = set()
    for i in tqdm(range(num_questions), desc="질문 생성 중"):
        chain = random_question_chain()
        question = chain.invoke({}).strip()
        if question.startswith("```") and question not in seen:
            print(f"\n생성된 질문 {i + 1}:\n{question}\n")
            questions.append(question)
            seen.add(question)
    return questions

# 질문에 DuckDuckGo + GPT로 답변 생성
def generate_with_rag(question):
    return rag_chain.invoke(question).strip()

# 전체 평가 및 저장
def create_and_save_dataset():
    generated_data = []

    # 1. 질문 생성
    questions = generate_questions_from_gpt(num_questions=1000)

    # 2. 질문마다 DuckDuckGo + GPT 답변 생성
    for idx, question in enumerate(tqdm(questions, desc="질문 처리 중")):
        print(f"\n질문 {idx + 1}: {question}")
        try:
            gpt_answer = generate_with_rag(question)
        except Exception as e:
            gpt_answer = f"오류 발생: {e}"
        print(f"GPT 응답:\n{gpt_answer}")

        generated_data.append({
            "question": question,
            "real_answer": gpt_answer,
            "label": 0,
        })

        time.sleep(1.2)  # DuckDuckGo rate limit 대응

    # 3. JSON 저장
    with open('generated_data.json', 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

    # 4. CSV 저장
    with open('generated_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "real_answer", "label"])
        writer.writeheader()
        writer.writerows(generated_data)

    print("\nDuckDuckGo 기반 데이터 생성 완료!")


# 실행
if __name__ == "__main__":
    create_and_save_dataset()
