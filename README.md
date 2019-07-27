# Speech-to-Text
This repository contains python code to convert audio to text. This system is codded using API as well as without using API. 
## Using API
### User Documentation
#### Installation
Using Speech Recognition API.
Instalation details for **speech_recognition** can be found [here](https://pypi.org/project/SpeechRecognition/).
#### Result
Following is the result you will get after successful compilation of code.
![Image of Result](https://github.com/San-B-09/Speech-to-Text/blob/master/Using_API/Result_Speech_Recognition_API.png)
### Developer Documentation
#### Algorithm
1.	SpeechRecognition library is imported on python console.
2.	Recognizer is initialized.
3.	With source as microphone, the recognizer uses 'listen' function to take the input of speech or audio files.
4.	This is stored in a variable named audio.
5.	The open and write function return a byte string representing the contents of a WAV file containing the audio.
6.	recognizer_google () is used to convert the audio file into text.
7.	The result in text format is printed.
#### Handling Exception
Audio that cannot be matched to text by the API, raises an UnknownValueError exception.

## Without using API
#### Collection Dataset
The dataset used for the project is downloaded from tensorflow. To download dataset [click here](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
#### Installation
The main modules used for the project are: 
* **scipy** (For download details [click here](https://www.scipy.org/install.html))
* **keras** (For download details [click here](https://keras.io/#installation))
* **pyaudio** (For download details [click here](https://pypi.org/project/PyAudio/))
* **pydub** (For download details [click here](https://pypi.org/project/pydub/))
* **glob** (For download details [click here](https://pypi.org/project/glob2/))

