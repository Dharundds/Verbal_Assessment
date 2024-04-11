import json
import os,sys
import pickle
import warnings
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler, normalize
from functions.get_relevance import LLMModel
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
import random
## Downsamlping and converting mp3 to wav format
# def convert(filePath):
    # wavFile="./media/6313.mp3"
    # wav,_ = librosa.load(filePath,sr=16000)
    # sf.write("./media/6313.wav", wav, 16000, 'PCM_16')


def VoiceActivityDetection(wavData, frameRate):
    # uses the librosa library to compute short-term energy
    ste = librosa.feature.rms(y=wavData,hop_length=int(16000/frameRate)).T
    thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
    return (ste>thresh).astype('bool')

def trainGMM(wavFile, frameRate, segLen, vad, numMix):
    wavData,_ = librosa.load(wavFile,sr=16000)
    mfcc = librosa.feature.mfcc(y=wavData, sr=16000, n_mfcc=20,hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad,:]
    print("Training GMM..")
    GMM = GaussianMixture(n_components=numMix,covariance_type='diag').fit(mfcc)
    var_floor = 1e-5
    segLikes = []
    segSize = frameRate*segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0])/(frameRate*segLen)))):
        startI = segI*segSize
        endI = (segI+1)*segSize
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0]-1
        if endI==startI:    # Reached the end of file
            break
        seg = mfcc[startI:endI,:]
        compLikes = np.sum(GMM.predict_proba(seg),0)
        segLikes.append(compLikes/seg.shape[0])
    print("Training Done")
    return np.asarray(segLikes)

def SegmentFrame(clust, segLen, frameRate, numFrames):
    frameClust = np.zeros(numFrames)
    for clustI in range(len(clust)-1):
        frameClust[clustI*segLen*frameRate:(clustI+1)*segLen*frameRate] = clust[clustI]*np.ones(segLen*frameRate)
    frameClust[(clustI+1)*segLen*frameRate:] = clust[clustI+1]*np.ones(numFrames-(clustI+1)*segLen*frameRate)
    return frameClust


def speakerdiarisationdf(hyp, frameRate, wavFile):
    audioname=[]
    starttime=[]
    endtime=[]
    speakerlabel=[]
    relevance = 0
    try:
        spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]
        if spkrChangePoints[0]!=0 and hyp[0]!=-1:
            spkrChangePoints = np.concatenate(([0],spkrChangePoints))
        spkrLabels = []
        for spkrHomoSegI in range(len(spkrChangePoints)):
            spkrLabels.append(hyp[spkrChangePoints[spkrHomoSegI]+1])
        count = 0
        for spkrI,spkr in enumerate(spkrLabels[:-1]):
            if spkr!=-1:
                audioname.append(wavFile.split('/')[-1].split('.')[0]+".wav")
                starttime.append((spkrChangePoints[spkrI]+1)/float(frameRate))
                if count <2:
                    print("change point",spkrChangePoints[spkrI]+1)
                    print(float(frameRate),end="\n\n")

                    count +=1
                endtime.append((spkrChangePoints[spkrI+1]-spkrChangePoints[spkrI])/float(frameRate))
                # speakerlabel.append("Speaker "+str(int(spkr)))
                if int(spkr) == 0:  # Check if it's the second speaker
                    speakerlabel.append("Speaker 2")
                else:
                    speakerlabel.append("Speaker 1")
        if spkrLabels[-1]!=-1:
            audioname.append(wavFile.split('/')[-1].split('.')[0]+".wav")
            starttime.append(spkrChangePoints[-1]/float(frameRate))
            endtime.append((len(hyp) - spkrChangePoints[-1])/float(frameRate))
            speakerlabel.append("Speaker "+str(int(spkrLabels[-1])))

        rel = random.randint(20,80)
        speakerdf=pd.DataFrame({"Audio":audioname,"starttime":starttime,"endtime":endtime,"speakerlabel":speakerlabel,"relevance":rel})

        spdatafinal=pd.DataFrame(columns=['Audio','SpeakerLabel','StartTime','EndTime',"relevance"])
        i=0
        k=0
        j=0
        spfind=""
        stime=""
        etime=""
        for row in speakerdf.itertuples():
            rel = random.randint(40,80)
            if(i==0):
                spfind=row.speakerlabel
                stime=row.starttime
            else:
                if(spfind==row.speakerlabel):
                    etime=row.starttime
                else:
                    spdatafinal.loc[k]=[wavFile.split('/')[-1].split('.')[0]+".wav",spfind,stime,row.starttime,rel]
                    k=k+1
                    spfind=row.speakerlabel
                    stime=row.starttime
            i=i+1
        spdatafinal.loc[k]=[wavFile.split('/')[-1].split('.')[0]+".wav",spfind,stime,etime,rel]
        return spdatafinal
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        
def speakerdiarisation_optimized(hyp, frameRate, wavFile):
    audioname = []
    starttime = []
    endtime = []
    speakerlabel = []

    spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]
    if spkrChangePoints[0] != 0 and hyp[0] != -1:
        spkrChangePoints = np.concatenate(([0], spkrChangePoints))
    
    spkrLabels = [hyp[spkrChangePoints[i] + 1] for i in range(len(spkrChangePoints))]
    
    for spkrI, spkr in enumerate(spkrLabels[:-1]):
        if spkr != -1:
            audioname.append(wavFile.split('/')[-1].split('.')[0] + ".wav")
            starttime.append((spkrChangePoints[spkrI] + 1) / float(frameRate))
            endtime.append((spkrChangePoints[spkrI + 1] - spkrChangePoints[spkrI]) / float(frameRate))
            speakerlabel.append("Speaker 2" if int(spkr) == 0 else "Speaker 1")
    
    if spkrLabels[-1] != -1:
        audioname.append(wavFile.split('/')[-1].split('.')[0] + ".wav")
        starttime.append(spkrChangePoints[-1] / float(frameRate))
        endtime.append((len(hyp) - spkrChangePoints[-1]) / float(frameRate))
        speakerlabel.append("Speaker " + str(int(spkrLabels[-1])))

    speakerdf = pd.DataFrame({"Audio": audioname, "starttime": starttime, "endtime": endtime, "speakerlabel": speakerlabel})

    spdatafinal = pd.DataFrame(columns=['Audio', 'SpeakerLabel', 'StartTime', 'EndTime'])
    k = 0
    spfind = ""
    stime = ""
    etime = ""
    
    for i, row in enumerate(speakerdf.itertuples()):
        if i == 0:
            spfind = row.speakerlabel
            stime = row.starttime
        else:
            if spfind == row.speakerlabel:
                etime = row.starttime
            else:
                duration = etime - stime
                text_transcript = getRelevancyScore("./media/6313.wav", stime, etime)
                spdatafinal.loc[k] = [wavFile.split('/')[-1].split('.')[0] + ".wav", spfind, stime, row.starttime]
                k += 1
                spfind = row.speakerlabel
                stime = row.starttime

    spdatafinal.loc[k] = [wavFile.split('/')[-1].split('.')[0] + ".wav", spfind, stime, etime]
    return spdatafinal

def getRelevancyScore(audio, start_time, end_time):
    r = sr.Recognizer()
    
    with sr.AudioFile(audio) as source:
        audio = r.record(source, offset=start_time, duration=end_time - start_time)
        try:
            text = r.recognize_google(audio)
            print("Text: " + text)
        except Exception as e:
            print("Exception: " + str(e))
    return True
    


def transcribe_segments_concurrently(audio, segments):
    with ThreadPoolExecutor() as executor:
        results = executor.map(getRelevancyScore, [audio] * len(segments), segments)
    return list(results)




def main(wavFile):
    try:
        # wavFile = f".{wavFile}"
        print(wavFile)
        segLen,frameRate,numMix = 3,50,128
        wavData,_ = librosa.load(wavFile,sr=16000)
        vad=VoiceActivityDetection(wavData,frameRate)
        clusterset = trainGMM(wavFile, frameRate, segLen, vad, numMix)
        vad=VoiceActivityDetection(wavData,frameRate)

        scaler = StandardScaler()
        # Scaling the data so that all the features become comparable
        X_scaled = scaler.fit_transform(clusterset)
        # Normalizing the data so that the data approximately
        # follows a Gaussian distribution
        X_normalized = normalize(X_scaled)
        cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
        clust=cluster.fit_predict(X_normalized)
        mfcc = librosa.feature.mfcc(y=wavData, sr=16000, n_mfcc=20,hop_length=int(16000/frameRate)).T
        frameClust = SegmentFrame(clust, segLen, frameRate, mfcc.shape[0])

        pass1hyp = -1*np.ones(len(vad))
        np.putmask(pass1hyp, vad, frameClust)

        spkdf=speakerdiarisationdf(pass1hyp, frameRate, wavFile)
        # spkdf=speakerdiarisation_optimized(pass1hyp, frameRate, wavFile)
        
        spkdf_dict = spkdf.to_dict(orient='records')
        # llm = LLMModel()
        # status, resp = llm.getFromModel(spkdf_dict)
        # if status:
        #     resp = resp.strip().split("json")[1].strip()
        #     resp = resp.strip().split("\n")[0].strip()
        #     return resp
        # else:
        #     return []
        return spkdf_dict
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return {}

if __name__ == "__main__":
    print(main("./media/6313.mp3"))