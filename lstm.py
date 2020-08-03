# Build and train LSTM part of jaki. If required also contains functions for one-hot encoding of drum loops from
#  BFD format

import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd

oneHotPatterns = np.load("one_hot_drum_loops.npy")
np.random.shuffle(oneHotPatterns)

print("Number of Patterns = ", oneHotPatterns.shape[0])

bar1 = oneHotPatterns[0:5000,0:16,:]
bar2 = oneHotPatterns[0:5000,16:32,:]

bar1Validate = oneHotPatterns[5000:oneHotPatterns.shape[0],0:16,:]
bar2Validate = oneHotPatterns[5000:oneHotPatterns.shape[0],16:32,:]

X_train = bar1
y_train = bar2
X_valid = bar1Validate
y_valid = bar2Validate


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(250, input_shape=(16,32),return_sequences=True)) #encoder
model.add(tf.keras.layers.LSTM(100, input_shape=(16,32),dropout=0.2,recurrent_dropout=0.2)) #encoder
model.summary()

model.add(tf.keras.layers.RepeatVector(16))
model.add(tf.keras.layers.LSTM(16, return_sequences=True, input_shape=(100,))) #decoder
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='softmax')))

model.compile(
     loss='mse', metrics=['accuracy'])

model.summary()

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)]

#x_train = first bar, y_train = second bar
history = model.fit(X_train,  y_train,
                    batch_size=256, epochs=2000,
                    callbacks=callbacks,
                    validation_data=(bar1Validate,bar2Validate))

model.save("JAKI_Encoder_Decoder_3-8_1")
