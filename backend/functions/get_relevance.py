import os
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import json
# from IPython.display import Markdown

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
class LLMModel:
    def __init__(self):
        genai.configure(api_key="AIzaSyALF6do0EC4l-iG5iREB48CLDaYyTtdiNg")
        print("started file upload")
        self.file = self.uploadFile()
        print("file upload completed")

    def uploadFile(self):
        file = genai.upload_file(path="./media/6313.mp3",display_name="test-audio")
        print(f"file uploaded '{file.name}' as: {file.uri}")
        return file

    def getFromModel(self,jsons):
        try:
            isFile = genai.get_file(name=self.file.name)
            if isFile.name:
                
                model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
                # print(jsons)
                response = model.generate_content([f"""go throught this file and  this is after speaker diaziation {json.dumps(jsons)}
                                                go through this list of jsons and give the relevance score on a scale of 100 the text of the given topic and only give the score and topic(generalized one) in the mentioned format add relevancy score to that list json i give and give on the same format  on each interval between the start and end time in the each json
                                                    , only give only on the mentioned  json format in single line, only give the list of jsons alone remove the triple single quote in start and end give only list of jsons""", self.file])
                
                # print(response.text)
                # genai.delete_file(self.file.name)
                # print(f'Deleted {self.file.display_name}.')

            return True, response.text
        except Exception as e:
            return False,"No file exists"


if __name__ == "__main__":
    llm = LLMModel()
    status, resp = llm.sendToModel()
    # sendToModel()