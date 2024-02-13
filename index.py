# 토큰 정보 로드를 위한 라이브러리
# 설치: pip install python-dotenv
from dotenv import load_dotenv
import os

#토큰 정보 로드
load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "Runnables"

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()


# 1. 질문을 입력받는다
# 2. 프롬프트로 질문이 들어간다
# 3. 프롬프트를 모델로 전달한다
# 4. 결과를 출력한다

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{country}의 수도가 어디고, {capital}의 날씨 알려줘")
prompt 

from langchain_openai import ChatOpenAI

model = ChatOpenAI(openai_api_key="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")

chain = prompt | model | output_parser

type_check_str = chain.invoke({"country": "대한민국", "capital" : "서울"})

type_check_map = {"Answer" : type_check_str}
print(type(type_check_str))
print(type_check_str)
print(type(type_check_map))