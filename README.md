# jaki

Work in progress.

jaki is and automatic drum loop generation system, built to generate controlled, musical variations on 1 bar 5-part drum loops.

The first part of the system uses a LSTM Encoder-Decoder architecture to learn to generate 1 bar loops based on a 1 bar seed loop. The system is fed with a dataset of 2 bar loops, and learns to predict the second bar from the first bar, thus generating continuations of a 1 bar loop that are idiomatic. This architecture is inspired by the RL Tuner algorithm developed by Magenta for melody generation. 

The output of the LSTM can often be either musically uninteresting, or sometimes flawed in the case of atypical seed loops. A common problem is excessive repetition of certain hits, and a general lack of complexity in the generated loop. The second part of jaki therefore uses a Deep-Q reinforcement learning architecture to 'tune' the output of the LSTM to more musically interesting and complex loops. The reward function calculates complexity using two musical features (taken from my GrooveToolbox implementations) - density and syncopation. The reward function is written to target a higher density (more onsets in the loop), but not too high a density, and increased syncopation in the generated loop. 

To connect to the LSTM, the DQN reward function is calculated based on closeness to the probability distribution predicted by the LSTM. The weights of the DQN are also initialised to this, following the approach used by Magenta to connect a their Note RNN to a DQN.

As of 20-6-20 the system currently works in a basic sense, but significant hyperparameter tuning is required. The LSTM model is mostly complete, but the DQN requires work, and a better look at the reward calculation in particular.

As further work I hope to look at the incorporation of more musical features to be used in the reward function, with adaptable weightings, aiming towards and interactive drum loop variation generator with higher-level musical controls. As part of developing the reinforcement learning aspect of jaki, I hope to look at more developed ways of integrating multiple reward functions and action selection, such as option-based frameworks and deep hierarchical reinforcement learning.
