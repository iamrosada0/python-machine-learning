import tkinter as tk
from tkinter import messagebox, scrolledtext  # Import scrolledtext here
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

# Load data (limited to 500 rows for testing)
df = pd.read_excel("portuguese.xlsx", names=["English", "Portuguese"], nrows=500)
print(df.head())  # To see the first 5 rows

# Tokenize English and Portuguese (Model 1: English to Portuguese, Model 2: Portuguese to English)
english_texts = df["English"].values
portuguese_texts = df["Portuguese"].values

# Tokenizer 1 (English to Portuguese)
english_tokenizer1 = Tokenizer()
portuguese_tokenizer1 = Tokenizer()

# Tokenizer 2 (Portuguese to English)
english_tokenizer2 = Tokenizer()
portuguese_tokenizer2 = Tokenizer()

# Fit on English and Portuguese texts
english_tokenizer1.fit_on_texts(english_texts)
portuguese_tokenizer1.fit_on_texts(portuguese_texts)

portuguese_tokenizer2.fit_on_texts(portuguese_texts)
english_tokenizer2.fit_on_texts(english_texts)

# Check if 'eu' is in the right tokenizer
word = "eu"

# Check in Portuguese tokenizer 1 (English to Portuguese)
if word in portuguese_tokenizer1.word_index:
    print(f"'{word}' is in the vocabulary of Portuguese Tokenizer 1 (English to Portuguese) with index {portuguese_tokenizer1.word_index[word]}")
else:
    print(f"'{word}' is NOT in the vocabulary of Portuguese Tokenizer 1 (English to Portuguese).")

# Check in Portuguese tokenizer 2 (Portuguese to English)
if word in portuguese_tokenizer2.word_index:
    print(f"'{word}' is in the vocabulary of Portuguese Tokenizer 2 (Portuguese to English) with index {portuguese_tokenizer2.word_index[word]}")
else:
    print(f"'{word}' is NOT in the vocabulary of Portuguese Tokenizer 2 (Portuguese to English).")

# Convert to sequences
english_sequences1 = english_tokenizer1.texts_to_sequences(english_texts)
portuguese_sequences1 = portuguese_tokenizer1.texts_to_sequences(portuguese_texts)

english_sequences2 = english_tokenizer2.texts_to_sequences(portuguese_texts)
portuguese_sequences2 = portuguese_tokenizer2.texts_to_sequences(english_texts)

# Pad sequences
max_len_english = max(len(seq) for seq in english_sequences1)
max_len_portuguese = max(len(seq) for seq in portuguese_sequences1)

english_padded1 = pad_sequences(english_sequences1, maxlen=max_len_english, padding="post")
portuguese_padded1 = pad_sequences(portuguese_sequences1, maxlen=max_len_portuguese, padding="post")

# Add special tokens for the decoder in Model 1 (English to Portuguese)
start_token = portuguese_tokenizer1.word_index.get("<start>", len(portuguese_tokenizer1.word_index) + 1)
end_token = portuguese_tokenizer1.word_index.get("<end>", len(portuguese_tokenizer1.word_index) + 2)

portuguese_tokenizer1.word_index["<start>"] = start_token
portuguese_tokenizer1.word_index["<end>"] = end_token
portuguese_tokenizer1.index_word[start_token] = "<start>"
portuguese_tokenizer1.index_word[end_token] = "<end>"

portuguese_padded1 = np.hstack(
    [
        np.full((portuguese_padded1.shape[0], 1), start_token),
        portuguese_padded1,
    ]
)

# Pad sequences for Model 2 (Portuguese to English)
max_len_english2 = max(len(seq) for seq in english_sequences2)
max_len_portuguese2 = max(len(seq) for seq in portuguese_sequences2)

english_padded2 = pad_sequences(english_sequences2, maxlen=max_len_english2, padding="post")
portuguese_padded2 = pad_sequences(portuguese_sequences2, maxlen=max_len_portuguese2, padding="post")

# Add special tokens for the decoder in Model 2 (Portuguese to English)
start_token2 = english_tokenizer2.word_index.get("<start>", len(english_tokenizer2.word_index) + 1)
end_token2 = english_tokenizer2.word_index.get("<end>", len(english_tokenizer2.word_index) + 2)

english_tokenizer2.word_index["<start>"] = start_token2
english_tokenizer2.word_index["<end>"] = end_token2
english_tokenizer2.index_word[start_token2] = "<start>"
english_tokenizer2.index_word[end_token2] = "<end>"

english_padded2 = np.hstack(
    [
        np.full((english_padded2.shape[0], 1), start_token2),
        english_padded2,
    ]
)
# Define Encoder Model 1 for English
encoder_inputs1 = Input(shape=(None,))
encoder_embedding1 = Embedding(input_dim=len(english_tokenizer1.word_index) + 1, output_dim=256)(encoder_inputs1)
encoder_lstm1 = LSTM(256, return_state=True)
encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_embedding1)
encoder_states1 = [state_h1, state_c1]
encoder_model1 = Model(encoder_inputs1, encoder_states1)

# Define Decoder Model 1 for English to Portuguese
decoder_inputs1 = Input(shape=(None,))
decoder_embedding1 = Embedding(input_dim=len(portuguese_tokenizer1.word_index) + 1, output_dim=256)(decoder_inputs1)
decoder_lstm1 = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs1, state_h1, state_c1 = decoder_lstm1(decoder_embedding1, initial_state=encoder_states1)
decoder_dense1 = TimeDistributed(Dense(len(portuguese_tokenizer1.word_index) + 1, activation="softmax"))
decoder_outputs1 = decoder_dense1(decoder_outputs1)
decoder_model1 = Model([decoder_inputs1] + encoder_states1, [decoder_outputs1, state_h1, state_c1])

# Define Encoder Model 2 for Portuguese
encoder_inputs2 = Input(shape=(None,))
encoder_embedding2 = Embedding(input_dim=len(portuguese_tokenizer2.word_index) + 1, output_dim=256)(encoder_inputs2)
encoder_lstm2 = LSTM(256, return_state=True)
encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_embedding2)
encoder_states2 = [state_h2, state_c2]
encoder_model2 = Model(encoder_inputs2, encoder_states2)

# Define Decoder Model 2 for Portuguese to English
decoder_inputs2 = Input(shape=(None,))
decoder_embedding2 = Embedding(input_dim=len(english_tokenizer2.word_index) + 1, output_dim=256)(decoder_inputs2)
decoder_lstm2 = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs2, state_h2, state_c2 = decoder_lstm2(decoder_embedding2, initial_state=encoder_states2)
decoder_dense2 = TimeDistributed(Dense(len(english_tokenizer2.word_index) + 1, activation="softmax"))
decoder_outputs2 = decoder_dense2(decoder_outputs2)
decoder_model2 = Model([decoder_inputs2] + encoder_states2, [decoder_outputs2, state_h2, state_c2])
# Define the LSTM model for translation (Model 1: English to Portuguese)
latent_dim = 64

# Encoder
encoder_inputs1 = Input(shape=(max_len_english,))
encoder_embedding1 = Embedding(len(english_tokenizer1.word_index) + 1, latent_dim)(encoder_inputs1)
_, state_h1, state_c1 = LSTM(latent_dim, return_state=True)(encoder_embedding1)
encoder_states1 = [state_h1, state_c1]

# Decoder
decoder_inputs1 = Input(shape=(None,))
decoder_embedding1 = Embedding(len(portuguese_tokenizer1.word_index) + 1, latent_dim)(decoder_inputs1)
decoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs1, _, _ = decoder_lstm1(decoder_embedding1, initial_state=encoder_states1)
decoder_dense1 = TimeDistributed(Dense(len(portuguese_tokenizer1.word_index) + 1, activation="softmax"))
decoder_outputs1 = decoder_dense1(decoder_outputs1)

# Compile the model
model1 = Model([encoder_inputs1, decoder_inputs1], decoder_outputs1)
model1.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy")

# Prepare target data for training (Model 1)
decoder_input_data1 = portuguese_padded1[:, :-1]  # Portuguese input sequences (shifted by one)
decoder_target_data1 = portuguese_padded1[:, 1:]  # Portuguese target sequences (shifted by one)

# Train the model
model1.fit(
    [english_padded1, decoder_input_data1],
    decoder_target_data1[..., np.newaxis],  # Expand dimensions to match output shape
    batch_size=2,
    epochs=1,
)

# GUI Setup
root = tk.Tk()
root.title("Bidirectional Translation Tool")

# Create a scrolled text widget for the translation result
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
output_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
def decode_sentence(input_seq, encoder_model, decoder_model, tokenizer, reverse_word_index, max_decoder_seq_length):
    # Initialize the sequence with the start token
    start_token = tokenizer.word_index["<start>"]
    stop_token = tokenizer.word_index["<end>"]

    # Get the encoder's state from the input sequence
    states_value = encoder_model.predict(input_seq)

    # Create an empty target sequence, initialized with the start token
    target_seq = np.array([[start_token]])

    # Initialize an empty list to store the generated words
    translated_sentence = []

    for _ in range(max_decoder_seq_length):
        # Predict the next token using the decoder model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Get the most probable token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index[sampled_token_index]

        # If we hit the stop token, break
        if sampled_word == "<end>":
            break

        # Append the word to the translated sentence
        translated_sentence.append(sampled_word)

        # Update the target sequence (add the predicted word to the target sequence)
        target_seq = np.array([[sampled_token_index]])

        # Update the decoder's states
        states_value = [h, c]

    # Join the list of translated words into a sentence
    return " ".join(translated_sentence)

# Reverse word index for Portuguese (after training the tokenizer)
reverse_portuguese_word_index1 = {value: key for key, value in portuguese_tokenizer1.word_index.items()}
reverse_english_word_index2 = {value: key for key, value in english_tokenizer2.word_index.items()}

def translate():
    mode = mode_var.get()
    input_text = input_entry.get("1.0", tk.END).strip()  # Get the text input from the user
    
    if mode == "1":  # English to Portuguese
        input_sequence = english_tokenizer1.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_sequence, maxlen=max_len_english, padding="post")
        translated_sentence = decode_sentence(
            input_padded,
            encoder_model1,
            decoder_model1,
            portuguese_tokenizer1,
            reverse_portuguese_word_index1,
            max_len_portuguese
        )
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, translated_sentence)

    elif mode == "2":  # Portuguese to English
        input_sequence = portuguese_tokenizer2.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_sequence, maxlen=max_len_portuguese2, padding="post")
        translated_sentence = decode_sentence(
            input_padded,
            encoder_model2,
            decoder_model2,
            english_tokenizer2,
            reverse_english_word_index2,
            max_len_english2
        )
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, translated_sentence)
# Create widgets
mode_var = tk.StringVar(value="1")  # Default to English to Portuguese
english_to_portuguese_radio = tk.Radiobutton(root, text="English to Portuguese", variable=mode_var, value="1")
portuguese_to_english_radio = tk.Radiobutton(root, text="Portuguese to English", variable=mode_var, value="2")
input_label = tk.Label(root, text="Enter text to translate:")
input_entry = tk.Text(root, height=5, width=50)
translate_button = tk.Button(root, text="Translate", command=translate)

# Layout
input_label.grid(row=0, column=0, padx=10, pady=5)
input_entry.grid(row=1, column=0, padx=10, pady=5)
english_to_portuguese_radio.grid(row=2, column=0, padx=10, pady=5)
portuguese_to_english_radio.grid(row=2, column=1, padx=10, pady=5)
translate_button.grid(row=3, column=0, columnspan=2, pady=10)
root.mainloop()
