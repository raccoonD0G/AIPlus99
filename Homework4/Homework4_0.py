import os
import bs4
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.docstore.document import Document

from bs4 import BeautifulSoup
import requests

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key="")

# DuckDuckGo 기반 검색
search_tool = DuckDuckGoSearchRun()

def duckduckgo_retriever(question: str) -> list[str]:
    result = search_tool.run(question)
    return [result]

# Web 문서 로드
class KoreanWebLoader(WebBaseLoader):
    def _scrape(self, url, bs_kwargs=None):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, "html.parser")
        return soup


    def _parse(self, html):
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.find_all(class_="editedContent")
        texts = [e.get_text(separator="\n", strip=True) for e in elements if e.get_text(strip=True)]
        return [Document(page_content="\n\n".join(texts))]

loader = KoreanWebLoader(["https://spartacodingclub.kr/blog/all-in-challenge_winner"])
docs = loader.load()

# 벡터 분할 및 임베딩
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(api_key="")
)
chroma_retriever = vectorstore.as_retriever()

# 최종 retriever: DuckDuckGo + Chroma 결과 합치기
def hybrid_retriever(question: str) -> dict:
    ddg_result = duckduckgo_retriever(question)
    chroma_result = chroma_retriever.invoke(question)
    chroma_text = "\n\n".join(doc.page_content for doc in chroma_result)

    return {
        "context_ddg": "\n\n".join(ddg_result),
        "context_chroma": chroma_text,
        "question": question
    }

# 프롬프트
prompt_template = PromptTemplate.from_template("""
너는 정보 요약 전문가야. 아래 두 가지 출처(실시간 웹 검색, 특정 블로그)에서 추출한 내용을 참고해서 질문에 답변해줘.
추측하지 말고, 아래 문맥을 기반으로 정확하게 대답해.

[실시간 DuckDuckGo 검색 결과]
{context_ddg}

[블로그 분석 결과]
{context_chroma}

[질문]
{question}
""")

# 전체 파이프라인
rag_chain = (
    RunnableLambda(hybrid_retriever)
    | prompt_template
    | llm
    | StrOutputParser()
)

# 실행
user_msg = "ALL-in 코딩 공모전 수상작들을 요약해줘."

retrieved = hybrid_retriever(user_msg)

print("[DuckDuckGo 검색 결과 요약]:\n")
print(retrieved["context_ddg"])
print("\n" + "="*100 + "\n")

print("[Chroma (블로그) 추출 문서 내용]:\n")
print(retrieved["context_chroma"])
print("\n" + "="*100 + "\n")


response = rag_chain.invoke(user_msg)

print("응답:\n")
print(response)
