# jaki
https://zenodo.org/record/4285414#.YBgUydzLdPY

jaki is an automatic drum pattern generation system, built to generate controlled, musical continuations of 1 bar 5-part drum patterns (kick, snare, hihat, crash, tom).

The first part of the system uses a LSTM Encoder-Decoder architecture to learn to generate 1 bar patterns based on a 1 bar seed pattern. The system is fed with a dataset of 2 bar patterns, and learns to predict the second bar from the first bar, thus generating continuations of a 1 bar pattern that are stylistically accurate. 

The output of the LSTM can often be either musically uninteresting, or sometimes flawed in the case of atypical seed patterns. A common problem is excessive repetition of certain hits, and a general lack of complexity in the generated pattern. The second part of jaki therefore uses a Deep-Q reinforcement learning architecture to 'tune' the output of the LSTM to more musically interesting patterns according to four musical features controlled by the user. This architecture is inspired by the RL Tuner algorithm developed by Magenta for melody generation. https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning

The four musical features (taken from the GrooveToolbox https://github.com/fredbru/GrooveToolbox) are cymbal density, drum density, syncopation and repetition (aka symmetry). When running the code, the user can set the target score of each of these four features to low, medium or high (0,1,2). Any combination of the four features may be used. The reward is calculated based on the closeness of feature scores of the generated pattern to the target scores. Thus jaki attempts to generate patterns that fit the qualities desired by the user.

The reward function also incorporates a similarity function calculated against the probability distribution predicted by the LSTM. This rewards changes to the pattern that are most probable according to the LSTM, and therefore ensures the patterns generated by the DQN fit the style of the input pattern as learned by the LSTM.

As further work we will be incorporating more musical features and looking at new methods of integrating multiple reward functions and action selection, such as option-based frameworks and deep hierarchical reinforcement learning.

A selection of examples with audio and MIDI rendering may be found at:
https://www.dropbox.com/sh/clo9gux5y5w8b3j/AACzp6GfGyFacWvSNq_-8PNNa?dl=0

This repository contains scripts for generating example patterns using the deep reinforcement learning algorithm, with a pre-trained model. It also contains functions for building your own training dataset from BFD format Groove files, and training the LSTM yourself, though this is not required for standard use.

# Files

run_dqn.py:
Main file for running jaki, containing Double Deep-Q learning algorithm for the tuner part. When run from terminal, provides a prompt for user input for each of the four features. For each, type 0,1 or 2 for low medium or high, or just leave blank (press enter) to skip that feature and not consider it. The tuner will then run for a random seed pattern from the training data (MIDI input support currently under development). Loads the pre-trained model for the LSTM. 

As it runs, it prints the current state of the pattern with rewards, seed pattern and LSTM generated pattern so you can keep track of it as it learns. Upon completion prints the final pattern in the terminal and saves the seed, LSTM generated loop and final DQN generated loop to seperate MIDI files.

lstm.py
Train LSTM part of jaki from .npy  format dataset (one_hot_drum_loops.npy). Present for reference but not required for standard use in generating loops due to the pretrained model (JAKI_Encoder_Decoder_14-6-20_2) being provided.

one_hot_drum_loops.npy: 
Dataset of around 5500 2-bar drum loops encoded in 5 part one-hot format. For use in LSTM training, also used in Run-DQN as a source of random seed patterns.

MIDI.py:
Utility functions for converting from list-based format to MIDI files.

extract_grooves_from_folder.py:
Generate seperate .npy files for relevant data in BFD3 format Groove files. Use this alongside create_training_data.py only if you would like to generate your own training data from BFD3 format Groove files.

create_training_data.py:
Make a dataset in same numpy format of one_hot_drum_loops.npy from a folder of individual numpy files using functions in GrooveToolbox. Use this only if you would like to build your own training dataset for the LSTM from a library.


For any questions, please feel free to get in touch at fred.bruford@gmail.com
