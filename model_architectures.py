from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, LayerNormalization

# For 34M param model
def build_model_large(vocab_size=255849, dropout=0.25, recurrent_dropout=0.25, 
                      output_dimension=128, maxlen=300):
    inputs = Input(shape=(maxlen,))
    x = Embedding(input_dim=vocab_size, output_dim=output_dimension, mask_zero=True)(inputs)
    x = LayerNormalization()(x)
    x = LSTM(256, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# For 6.6M param model
def build_model_small(vocab_size=50000, maxlen=300, embedding_dim=128, dropout=0.2, recurrent_dropout=0.2):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, mask_zero=True))
    model.add(LSTM(128, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(3, activation='softmax'))
    model.build(input_shape=(None, 300))  # Build with expected input shape
    return model
