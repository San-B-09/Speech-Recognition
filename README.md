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
### User Documentation
#### Collection Dataset
The dataset used for the project is downloaded from tensorflow. To download dataset [click here](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
#### Installation
The main modules used for the project are: 
* **scipy** (For download details [click here](https://www.scipy.org/install.html))
* **keras** (For download details [click here](https://keras.io/#installation))
* **pyaudio** (For download details [click here](https://pypi.org/project/PyAudio/))
* **pydub** (For download details [click here](https://pypi.org/project/pydub/))
* **Wave** (For download details [click here](https://docs.python.org/2/library/wave.html))
* **os, glob** (For download details [click here](https://pypi.org/project/glob2/))
* **NumPy** (For download details [click here](https://numpy.org/))
* **Matplotlib** (For download details [click here](https://matplotlib.org/))
* **pandas** (For download details [click here](https://pandas.pydata.org/))
* **Sklearn** (For download details [click here](https://scikit-learn.org/))
* **random** (For download details [click here](https://docs.python.org/3/library/random.html))

### Developer Documentation
#### Data acquisition
A dataset of 106,000 wav files were acquire from TensorFlow website. [Click here](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) to download.
It consists of 34 words which would be used for training the model.

#### Algorithm
1. Importing all the libraries utilised
2.	Defined a class ‘DatasetGenerator’ for operating on the dataset
3.	Reading data snippets and data visualisation
4.	Generating the dataframe
5.	Deep CNN model definition
6.	Training the model 
7.	Recording audio file from user
8.	Pre-Processing the acquired wav file
9.	Predicting text and finding accuracy score

#### Detailed explanation of working
* **Libraries**
  1.	os, glob – to get all paths inside the directory of ‘.wav’ format
  2.	NumPy – to convert audio files to NumPy array’s for ease in operation
  3.	Matplotlib – for data visualisation (to plot spectrogram)
  4.	pandas – for operating on the dataframe
  5.	random – for pre-processing audio files
  6.	SciPy – for reading audio file and data visualisation
  7.	Sklearn – for splitting dataset into train and test and calculating accuracy score
  8.	Keras – for yielding batches, to avoid overfitting, deep learning library (for CNN base model), to design layers in our model
  9.	PyAudio – to read audio from microphone
  10.	Wave – to save recorded ‘.wav’ files
  11.	Pydub – to pre-process recorded audio file and play them

* **Class *DatasetGenerator***
  1.	Constructor – to initialise labels and sample rate
  2.	text_to_lables() – to encode text to numeric for ease of operation
  3.	lables_to_text() – to decode text from labels for text output
  4.	load_data() – to load paths of audio file into dataframe.
  5.	apply_train_test_split() – to split dataframe into train and test data
  6.	apply_train_val_split() – to split dataframe into train and validation data
  7.	read_wav_file() – to read the wav files and normalise it
  8.	process_wav_file() – for pre-processing wav file removing the silent parts in it
  9.	generator() – it generates the test train and validation batches and yield it

*	**Data Visualisation**:
Data is visualised in the form of signal wave and spectrogram.

* **Dataframe Generation**:
Initialisation of labels, loading dataset into dataframe and splitting it into train, test and validation dataframe.


*	**CNN Model**:
Convolutional neural network is used to find patterns. We do that by convoluting over an audio and looking for patterns. In the first few layers of CNNs the network can identify length decibels and frequency, but we can then pass these patterns down through our neural net and start recognizing more complex features as we get deeper. This property makes CNN’s good at recognising speech.

  1. Pooling layer: It partitions the input audio into a set of non-overlapping chunks and, for each such sub-region, outputs a value. The intuition is that the exact location of a feature is less important than its rough location relative to other features. Max - Pooling outputs the maximum value of the sub-region.
  2. Batch Normalisation: We normalize the input layer by adjusting and scaling the activations. We should normalize them to speed up learning. Batch normalization reduces the amount by what the hidden unit values shift around (covariance shift).
  3. Convolutional 2D: It extracts features from a source audio and works on chunks depending on kernel size. Layers early in the network architecture learn fewer convolutional filters while layers deeper in the network will learn more filters.
  4. Dropout: Dropout is a technique where randomly selected neurons are ignored during training. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. As a neural network learns, neuron weights settle into their context within the network. Weights of neurons are tuned for specific features providing some specialization. Neighbouring neurons become to rely on this specialization, which if taken too far can result in a fragile model too specialized to the training data. This reliant on context for a neuron during training is referred to complex co-adaptations.
  5. Flatten: In between the convolutional layer and the fully connected layer, there is a ‘Flatten’ layer. Flattening transforms a two-dimensional matrix of features into a vector that can be fed into a fully connected neural network classifier.


*	**Training**:
For training model, batches size of 10 were trained on 15 epochs. Callbacks was used to avoid overfitting of model



*	**Recording audio**:
Audio was recorded into chunks and combined into a frame. It was also saved.


*	**Audio Pre-processing**:
Audio file containing more than one word is split into chunks of single word audio file. It detects silence region and truncates the silence according to requirements 

*	**Text prediction**:
It predicts labels on basis of recorded audio file. Accuracy score is 91.20556% for single word prediction


RESULTS




REFERENCES
i.	https://keras.io/
ii.	https://pypi.org/project/PyAudio/
iii.	https://pypi.org/project/pydub/
iv.	http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
v.	https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148
vi.	https://www.youtube.com/watch?v=RBgfLvAOrss
vii.	https://www.youtube.com/watch?v=WCUNPb-5EYI&t=1188s

