import time

import music21 as mc
import glob
import  numpy as np
from tensorflow.python.estimator import keras
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

"""
 getNotes :  - return a liste of notes  => [A2#, G3, F2-, [A3, B3, C3#] ..................]
             - to get the notes from the files we have in our mididata folder
"""
def getNotes():
    notes = []
    #print("enter getNote")
    for fileName in glob.glob("midiData/*.mid"):
        #each file.midi to format => <music21.stream.Score 0x7fd9712bba20>
        midiFile = mc.converter.parse(fileName)
        notes_to_parse = None

        try: # file has inst
            # partition look like <music21.stream.Score 0x7f28aaf954a8>
            partition = mc.instrument.partitionByInstrument(midiFile)
            # notesMIDI is a set of events (all played notes, chord and other informations)
            notesMIDI = partition.parts[0].recurse()
        except: # file has notes in a flat structure
            notesMIDI = midiFile.flat.notes

        for event in notesMIDI:
            if isinstance(event, mc.note.Note):
                #to get the pitch of every played note
                notes.append(str(event.pitch))
            elif isinstance(event, mc.chord.Chord):
                #played Chord, we have to encode id each note alone
                notes.append('.'.join(str(n) for n in event.normalOrder))

    #print("end getNote")
    return notes


"""
    getdata() : -return ( input[sequence of notes], output[note]) => ( [A3,B4,B3,C1#, .......] -----> [A5#] )
                -it's a function that changes string notes to an integer type, to process them before learning
"""
def getData(data,dataMaped,sq_len):

    #print("enter getDATA")
    network_X = []
    network_Y = []

    len_= len(data)-sq_len
    for i in range(0, len_, 1):
        network_X.append([dataMaped[key] for key in data[i:i + sq_len]])
        network_Y.append(dataMaped[data[i + sq_len]])

   # print("end getDATA")
    return  network_X,network_Y


"""
    getPreparedData():  return (listOf_input, listOf_Target)) 
                        using the retured lists for the trainig fase, after treat it (normalization,..)
"""

def getPreparedData(notes):

    #print("enter preparedDATA")
    # get all pitch names
    mapedNote = dict((element, num+1) for num, element in enumerate(sorted(set(item for item in notes))))

    sq_len = 100
    #sq_len = input("tap length of each sequence to predict the next note after it")
    len_ = len(notes) - sq_len

    # create input  and  outputs sequences :
    network_X, network_Y = getData(notes,mapedNote,sq_len)

    sizeInput = len(network_X)

    # reshape the input into a format compatible with LSTM layers - get the the transposed matrix of network_X
    network_X = np.reshape(network_X, (sizeInput, sq_len, 1))

    # data normalization
    network_X = network_X / float(len(set(notes)))
    network_Y = np_utils.to_categorical(network_Y)
    #print("end preparedDATA")
    return network_X,network_Y


def modelLSTM(netINPUT, notesAmount):
    # sequential model is linear stack of layers
    """create a Sequential by passing a list of layer instances """

    #print("enter modelLSTM")
    model = Sequential(
        [
            LSTM(
                512,
                input_shape=(netINPUT.shape[1], netINPUT.shape[2]),
                return_sequences=True
            ),
            Dropout(0.3),
            LSTM(512, return_sequences=True),
            Dropout(0.3),
            LSTM(512),
            Dense(256),
            Dropout(0.3),
            #number of neurals on the Layer before the last layer is equal to the number of notes (without occurrence)
            Dense(notesAmount),
            #Softmax is often used for classification in the output layer
            Activation('softmax')
        ]
    )

    #configure the learning proces, For a multi-class classification problem

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    #print("end modelLSTM")
    return model


def training(model, input, output):
    model.fit(input, output, epochs=200, batch_size=64)

def lunch():
    """ lunch """
    notes = getNotes()

    # get number of notes with eliminated occurence
    numberOfNotes = len(set(notes))

    _input, _output = getPreparedData(notes)

    model = modelLSTM(_input, numberOfNotes)
    print(_input.shape)
    print(_output.shape)
    print(numberOfNotes)
    training(model, _input, _output)

lunch()
