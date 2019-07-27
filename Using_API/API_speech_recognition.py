import speech_recognition as sr  # import the library

r = sr.Recognizer()  # initialize recognizer

with sr.Microphone() as source:  # mention source it will be either Microphone or audio files.
    print("Speak Anything :")
    audio = r.listen(source)  # listen to the source

with open("C:\\Users\\bijaw\\Desktop\\new_audio.wav", "wb") as file:
    file.write(audio.get_wav_data())

try:
    text = r.recognize_google(audio)  # use recognizer to convert our audio into text part.
    print("You said : {}".format(text))
except:
    print("Sorry could not recognize your voice")