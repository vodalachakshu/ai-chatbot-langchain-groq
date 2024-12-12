# main.py

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0.7, model_name="text-davinci-003")

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="You are a helpful AI assistant. {user_input}"
)

# Set up conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create LLMChain
chat_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)

def chatbot():
    print("Welcome to the AI Chatbot! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_chain.run(user_input=user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    chatbot()
