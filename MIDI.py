# UNFINISHED. Generate drum loop variations from just LSTM and write to MIDI files.

import numpy as np
import pretty_midi

def convertOneHotToList(groove):
    # convert one hot or probabilities matrix from LSTM model into a list format
    columns = ['c ', 'co', 'cot', 'ct', 'k ', 'kc', 'kco', 'kcot', 'kct', 'ko', 'kot',
               'ks', 'ksc', 'ksco', 'kscot', 'ksct', 'kso', 'ksot', 'kst', 'kt', 'o ',
               'ot', '--', 's ', 'sc', 'sco', 'scot', 'sct', 'so', 'sot', 'st', 't ']
    grooveList = []
    for i in range(groove.shape[0]):
        index = np.argmax(groove[i, :])
        grooveList.append(columns[index])
    return grooveList

def write_MIDI(name, loop):
    GMKeymap = {"kick": [35, 36], "snare": [37, 38, 40], "closed hihat": [42, 44], "open hihat": [46], "low tom": [41, 43, 45]}

    #LSTM = load_model("JAKI_Encoder_Decoder_14-6-20_2")
    #oneHotGrooves = np.load("One-Hot-Drum-Loops.npy")
    #oneBarGrooves = oneHotGrooves[:,0:16,:]

    midiloop = pretty_midi.PrettyMIDI()

    #loop = convertOneHotToList(oneHotGrooves[2,:,:])

    drumkit_program = pretty_midi.instrument_name_to_program('Cello')
    drumkit = pretty_midi.Instrument(program=drumkit_program)

    time = 0.0
    for i in range(len(loop)):
        if 'k' in loop[i]:
            note = pretty_midi.Note(velocity=100, pitch=35, start=time, end=time+0.125)
            drumkit.notes.append(note)
        if 'c' in loop[i]:
            note = pretty_midi.Note(velocity=100, pitch=42, start=time, end=time+0.125)
            drumkit.notes.append(note)
        if 'o' in loop[i]:
            note = pretty_midi.Note(velocity=100, pitch=46, start=time, end=time+0.125)
            drumkit.notes.append(note)
        if 't' in loop[i]:
            note = pretty_midi.Note(velocity=100, pitch=43, start=time, end=time+0.125)
            drumkit.notes.append(note)
        if 's' in loop[i]:
            note = pretty_midi.Note(velocity=100, pitch=37, start=time, end=time+0.125)
            drumkit.notes.append(note)
        try:
            drumkit.notes.append(note)
        except:
            pass
        time += 0.125 # semiquaver length at 120bpm

    midiloop.instruments.append(drumkit)

    midiloop.write(name)