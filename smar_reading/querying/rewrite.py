# querying/rewrite.py
# 职责：查询改写，替换掉一些指代词
from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from querying.query_prompt import REWRITE_PROMPT

class QueryRewriter:

    def __init__(self, llm:ChatTongyi) -> None:
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system",REWRITE_PROMPT),
             MessagesPlaceholder("chat_history"),
            ("human","原始问题：{question}\n请改为独立检索查询:")
        ])
        self.__llm = llm

    def rewrite(self,question:str,chat_history:List[BaseMessage]) -> str:
        q = (question or "").strip()
        if not q:
            return q

        if len(q) < 10 or not any(kw in q for kw in ["那篇", "刚才", "它", "这个", "该"]):
            return q

        try:
            chain = self.__prompt | self.__llm | StrOutputParser()
            rewrite = chain.invoke(input={"question":q,"chat_history":chat_history})
            return rewrite
        except Exception as e:
            return q
