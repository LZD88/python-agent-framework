from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatTongyi(model="qwen3-max")

prompt = ChatPromptTemplate.from_messages([
    ("system","请根据用户传入的商品ID去查询对应的商品是什么。[如果你无法查询或者查询不到，就直接告知用户你当前无法查询，不需要额外解释"),
    ("user","商品ID:{good_id}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke(input={"good_id":6742835}))