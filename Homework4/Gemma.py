from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.tools import DuckDuckGoSearchRun

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-2-2b-it").to("cuda")

search_tool = DuckDuckGoSearchRun()

def duckduckgo_retriever(question: str) -> list[str]:
    result = search_tool.run(question)
    # print("DuckDuckGo 검색 결과:\n", result)
    return [result]

retriever = RunnableLambda(duckduckgo_retriever)

def format_docs(docs: list[str]) -> str:
    return "\n\n".join(docs)

prompt = PromptTemplate.from_template("""
<|system|>
너는 정보 요약 전문가야. 아래 검색 결과만 참고해서 질문에 정확히 답해줘. 
추측하지 말고, 문맥 안에 있는 정보만 사용해.

[Search Context]
{context}

[Question]
{question}

<|assistant|>
""")

def run_gemma(prompt_obj) -> str:
    prompt = str(prompt_obj)
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**encoded, max_new_tokens=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1]

    return decoded.strip().strip("'").strip('"').lstrip("\n")


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | RunnableLambda(run_gemma)
)

user_msg = "GPT-4o는 언제 출시됐고 어떤 점이 좋아졌어?"
result = rag_chain.invoke(user_msg)
print("Gemma 응답:\n" + result)

