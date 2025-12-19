import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI

# 初始化 FastAPI
app = FastAPI(title="外贸AI产品问答")

# 全局变量：RAG 链（生产环境建议用缓存或懒加载）
_rag_chain = None

def get_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain
    
    try:
        # 1. 加载产品数据（从 products.csv）
        df = pd.read_csv("products.csv")
        documents = []
        for _, row in df.iterrows():
            text = (
                f"产品名称: {row.get('name', '')}\n"
                f"描述: {row.get('description', '')}\n"
                f"规格: {row.get('specs', '')}\n"
                f"价格(USD): {row.get('price_usd', '')}\n"
                f"起订量(MOQ): {row.get('moq', '')}"
            )
            documents.append(text)
        
        # 2. 初始化向量化
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        vectorstore = Chroma.from_texts(documents, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 3. 构建问答链
        prompt = ChatPromptTemplate.from_template(
            "你是一个专业的外贸销售助理，请根据以下产品信息回答客户问题。\n"
            "产品信息：{context}\n\n"
            "客户问题：{question}\n\n"
            "要求：\n"
            "- 如果信息不足，回答'请联系销售获取详细资料。'\n"
            "- 用简洁专业的中英文双语回答\n"
            "- 突出参数、价格、MOQ等关键信息"
        )
        model = ChatOpenAI(
            model="qwen-max",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            openai_api_key=os.getenv("sk-bb9dfa77a3834541bfe15ae228966388"),
            temperature=0.3
        )
        _rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return _rag_chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

class QuestionRequest(BaseModel):
    question: str

@app.post("/answer")
def answer_question(req: QuestionRequest):
    try:
        chain = get_rag_chain()
        result = chain.invoke(req.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
