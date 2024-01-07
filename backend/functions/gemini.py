import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# print(os.getenv("apikey"))

def getModel(key):
    """
        Get Model is used to return the gemini LLM  requires api-key
    """
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-pro")

    return model


def getRelevancyScore(model,context):
    """
        By passing the users speech returns relevancy score
    """



    res = model.generate_content(f"Please give the relevance score on a scale of 100 the text '{context}' of the given topic and only give the score and topic(generalized one) in the mentioned format '{{\"relevance_score\":<Relevance value>,\"topic\":<topic-name>}}'")

    return res.text

# print(res.text)