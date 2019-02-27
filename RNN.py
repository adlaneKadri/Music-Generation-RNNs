import music21 as mc
import glob
import time
import  numpy as np
#from keras.utils import np_utils
from tensorflow.python.keras.utils import np_utils

"""
 getNotes :  - return a liste of notes  => [A2#, G3, F2-, [A3, B3, C3#] ..................]
             - to get the notes from the files we have in our mididata folder
"""
def getNotes():
    notes = []

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

    return notes


"""
    getdata() : -return ( input[sequence of notes], output[note]) => ( [A3,B4,B3,C1#, .......] -----> [A5#] )
                -it's a function that changes string notes to an integer type, to process them before learning
"""
def getData(data,dataMaped,sq_len):

    network_X = []
    network_Y = []

    len_= len(data)-sq_len
    for i in range(0, len_, 1):
        network_X.append([dataMaped[key] for key in data[i:i + sq_len]])
        network_Y.append(dataMaped[data[i + sq_len]])

    return  network_X,network_Y


"""
    getPreparedData():  return (listOf_input, listOf_Target)) 
                        using the retured lists for the trainig fase, after treat it (normalization,..)
"""

def getPreparedData():
    notes = getNotes()

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
    print("Fin preparation données")
    return network_X,network_Y


getPreparedData()