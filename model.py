import numpy as np
embedding_matrix = np.random.random((len(word_index)+1, EMBEDDING)

for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
print('Length of embedding_matrix:', embedding_matrix.shape[0])


sequence_input = Input(shape=MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequece_input)
dense_1 = Dense(100, activation='tanh')(embedded_sequences)
max_pooling = GlobalMaxPoolingID()(dense_1)
dense_2 = Dense(2, activation='softmax')(max_pooling)
model = Model(sequence_input, dense_2)
model.complie(loss='categorical_crossentropy', optimizer='relu', metrics=['acc'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_val,y_val),nb_epoch=15, batch_size=50)
