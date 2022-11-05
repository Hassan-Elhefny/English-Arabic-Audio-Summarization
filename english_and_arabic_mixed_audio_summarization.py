# import libraries to get text from file and to making text language detection and visualization
from functools import cache
import numpy as np
from pydub import AudioSegment
import whisper
import streamlit as st
from pydub import AudioSegment
from googletrans import Translator
from summa.summarizer import summarize
import gtts 
import re
import numpy as np
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf



#------------------------------------------------------------------------------------------------------#
#add GPU to run with CPU
list_physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(list_physical_devices[0], True)
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
# initialize some parameters
translator = Translator()
translated_txt = ''
sumarize_text = ''

reg = re.compile(r'[a-zA-Z]')
reg2 = re.compile(r'[ا-ي]')
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
#function to build model and get mixed text from audio
@cache
def building_model_and_get_text(filename):
    model = whisper.load_model("small")
    result = model.transcribe(filename)
    return result['text']
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
#function to summarize text using text rank
def summarize_text_to_arabic(arabic_text, ratio):
    summarized_text = summarize(text = arabic_text, ratio= ratio, language= 'arabic')
    return summarized_text.strip()
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
# function to translate english words into arabic words
def translate_english_words_to_arabic(total_text):
    total_text = total_text.split()
    total_text_translated = ''  
    for i in range(0, len(total_text)):
        if reg2.findall(total_text[i]) != [] and reg.findall(total_text[i]) != []:
            print(reg.findall(total_text[i]))
            print("Before : "+total_text[i])
            arabic_characters = reg2.findall(total_text[i])
            for j in arabic_characters:
                total_text[i] = total_text[i].replace(j,'')
            print("After : "+total_text[i])
            total_text[i] = (translator.translate(total_text[i], dest='ar')).text
        elif reg.findall(total_text[i]) != []:
            total_text[i] = (translator.translate(total_text[i], dest='ar')).text   
    for i in total_text:
        total_text_translated = total_text_translated + " " + i
    return total_text_translated.strip()
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
# function to get the audio duration from input audio file
def get_audio_duration(file_name):
    data, sampling_rate = librosa.load(file_name)
    S = librosa.stft(data)
    duration = librosa.get_duration(S = S, sr = sampling_rate)
    return np.ceil(duration)
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
# function to save summarization text as audio
def save_sum_as_audio(sum_text):
    sum_file_mp3 = gtts.gTTS(sum_text, lang='ar', slow= False)
    sum_file_mp3.save("summarization.mp3")
    file_name = 'summarization.mp3'
    return file_name
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
def make_text_sentences(text):
  text_list = text.split()
  final_text = ''
  for i in range(len(text_list)):
    final_text = final_text +" "+ text_list[i]
    if i % 50 == 0 :
      final_text += '\n'
  return final_text
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
# function to show audio signal and some information about the input audio file like sampling rate, duration and so on...
def get_audio_information(file_name):
    data, sampling_rate = librosa.load(file_name)
    print("*************************************Audio Information**************************************")
    print("Sampling Rate is : {}".format(sampling_rate))
    print("Audio Duration is : {} Seconds".format(get_audio_duration(file_name)))
    print("*******************************************************************************************\n")
    st.write("""
            ### Sampling Rate is : {}
            ### Audio file Length = '{}' S,
            """.format(sampling_rate, get_audio_duration(file_name)))
    print("*************************************Audio Wave**********************************************")
    plt.figure(figsize=(10,5))
    librosa.display.waveshow(data, sr = sampling_rate)
    plt.title("Audio Wave")
    plt.tight_layout()
    st.header("Audio Wave")
    st.pyplot()
    print("********************************************************************************************\n")
    print("*************************************Audio Wave Information***********************************")
    plt.figure(figsize=(10,5))
    melspec = librosa.feature.melspectrogram(y= data, sr= sampling_rate)
    librosa.display.specshow(melspec, y_axis='mel', x_axis='time')
    plt.colorbar()
    plt.title("Audio Wave Information")
    plt.tight_layout()
    st.header("Audio Wave Information")
    st.pyplot()
    print("********************************************************************************************")
    print("*************************************Audio Wave MFCC***********************************")
    plt.figure(figsize=(10,5))
    mfcc = librosa.feature.mfcc(y= data, sr= sampling_rate)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("Audio Wave MFCC")
    plt.tight_layout()
    st.header("Audio Wave MFCC")
    st.pyplot()
    print("********************************************************************************************")
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
#streamlit header and initial padge parameters
st.set_page_config(page_icon="🎶", page_title="Summarize Mixed Audio", layout='wide', initial_sidebar_state='expanded')

st.header("Summarize Mixed Audio Files")
st.write('''
In this project we will summarize mixed audio files with **Arabic** and **English** language into audio file with single language 
**Arabic** or **English** contain the result of summarization in audio file.
''')

st.image(image='https://www.annlabmed.org/asset/images/sub/m-img-audio-banner2.jpg', caption="Summarization")

st.sidebar.header("Options")

audio_option = st.sidebar.selectbox("Audio File Option", ['Upload File', 'Add Mixed Text'])

st.set_option('deprecation.showPyplotGlobalUse', False)
#-------------------------------------------------------------------------------------------------------#



#-------------------------------------------------------------------------------------------------------#
#add functionalities to buttons
if audio_option == 'Upload File': 
    file = st.sidebar.file_uploader(label="Select File ...", type=['mp3','wav'], help='Please choose mp3 or wav files only.')
    if file is not None:
        with st.spinner("Please wait While collecting text from audio and summarization process end ...."):
            st.success("File Uploaded Successfully")
            st.subheader("Audio file name is **'{}'**".format(file.name))
            st.audio(file)
            st.balloons()
            
            audio_file_to_download = AudioSegment.from_mp3(file)
            audio_file_to_download.export(out_f = "./uploaded_file.mp3",
                        format = "mp3")
            
            st.write("""
            # Audio File Information And Audio Wave Visualization
            ### Audio file channels = '{}',
            """.format(audio_file_to_download.channels))

            get_audio_information('./uploaded_file.mp3')


            mixed_text = building_model_and_get_text('./uploaded_file.mp3')


            st.header("Captured Text From Audio")
            st.write(mixed_text)
            st.snow()
            translated_txt = translate_english_words_to_arabic(mixed_text)
            translated_txt = make_text_sentences(mixed_text)
            st.header("Text In Arabic Only After Translate English Words")
            st.write(translated_txt)
            st.header("Summarized Text")
            sumarized_text = summarize_text_to_arabic(translated_txt, 0.3)
            st.write(sumarized_text)
            st.balloons()
            st.sidebar.header("This Is The Summarization Audio File")
            sum_file_name = save_sum_as_audio(sumarized_text)
            st.sidebar.audio(sum_file_name)
            st.sidebar.snow()        
else:
    doc_text = st.text_input(label='Mixed Text', help='This is the place to pase the mixed document')
    if len(doc_text) > 50:
        with st.spinner("Please Wait While collecting Data From File ...."):
            translated_txt = translate_english_words_to_arabic(doc_text)
            translated_txt = make_text_sentences(translated_txt)
            st.header("Text In Arabic Only After Translate English Words")
            st.write(translated_txt)
            st.header("Summarized Text")
            sumarized_text = summarize_text_to_arabic(translated_txt, 0.3)
            st.write(sumarized_text)
            st.sidebar.header("This Is The Summarization Audio File")
            sum_file_name = save_sum_as_audio(sumarized_text)
            st.sidebar.audio(sum_file_name)
            st.sidebar.snow()
    else:
        time.sleep(5)
        st.error("Please Enter Text More Than 50 word")
#-------------------------------------------------------------------------------------------------------#