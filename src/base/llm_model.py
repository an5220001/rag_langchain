from langchain_google_genai import ChatGoogleGenerativeAI

def llm_model():
    api_key = 'AIzaSyD-AvK6_n6xzK3UEaRPxaXgLvXXKEcXlNk'
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key, temperature=0.3)
    return llm