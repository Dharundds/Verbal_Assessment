import os
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr

load_dotenv()

# print(os.getenv("apikey"))

def getModel(key):
    """
        Get Model is used to return the gemini LLM  requires api-key
    """
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-pro")

    return model


def getRelevancyScore(audio, segment):
    """
        By passing the users speech returns relevancy score
    """
    key = os.getenv("api_key")
    model = getModel(key)
    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio = r.record(source, offset=segment[0], duration=segment[1] - segment[0])
        try:
            text = r.recognize_google(audio)
            print("Text: " + text)
        except Exception as e:
            print("Exception: " + str(e))
    return True


    # res = model.generate_content(f"Please give the relevance score on a scale of 100 the text '{context}' of the given topic and only give the score and topic(generalized one) in the mentioned format '{{\"relevance_score\":<Relevance value>,\"topic\":<topic-name>}}'")

    # return res.text

# print(res.text)