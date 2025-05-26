# src/nlp/langchain_prompt.py
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import os

def generate_news_analysis(news_text):
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    prompt = PromptTemplate(
        input_variables=["news_text"],
        template="""
        Analyze the sentiment and classify the following news article as either 'Fake' or 'Real':
        {news_text}
        """
    )

    llm = OpenAI(temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(news_text=news_text)
