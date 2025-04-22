from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import os
os.environ["OPENAI_API_KEY"] = ""

search_tool = DuckDuckGoSearchRun()

def duckduckgo_retriever(question: str) -> list[str]:
    result = search_tool.run(question)
    # print("DuckDuckGo 검색 결과:\n", result)
    return [result]

retriever = RunnableLambda(duckduckgo_retriever)

def format_docs(docs: list[str]) -> str:
    return "\n\n".join(docs)

prompt = PromptTemplate.from_template("""
너는 정보 요약 전문가야. 아래 검색 결과만 참고해서 질문에 정확히 답해줘. 
추측하지 말고, 문맥 안에 있는 정보만 사용해.

[검색 결과]
{context}

[질문]
{question}
""")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

user_msg = "엘리자베스는 앨리스보다 키가 크고, 앨리스는 줄리아보다 키가 작다. 줄리아가 가장 키가 크다면, 누가 가장 키가 작을까?"
print("GPT 응답:")
print(rag_chain.invoke(user_msg))
