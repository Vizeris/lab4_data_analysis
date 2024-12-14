######################################################### 2

import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline
from PIL import Image
import requests

def translate_to_ukrainian(text):
    print("1. Translation from English to Ukrainian:")
    translator = pipeline("translation_en_to_uk", model="Helsinki-NLP/opus-mt-en-uk")
    translation = translator(text)
    print(f"Original Text: {text}\nTranslated Text: {translation[0]['translation_text']}\n")

def classify_image(image_url):
    print("2. Image Classification:")
    image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    image = Image.open(requests.get(image_url, stream=True).raw)
    results = image_classifier(image)
    print(f"Results for image '{image_url}':")
    for result in results:
        print(f"  - Label: {result['label']}, Score: {result['score']:.4f}")
    print()

def generate_text(prompt):
    print("3. Text Generation:")
    generator = pipeline("text-generation", model="gpt2")
    generated = generator(prompt, max_length=50, num_return_sequences=1)
    print(f"Prompt: {prompt}\nGenerated Text: {generated[0]['generated_text']}\n")

def extract_information(text, question):
    print("4. Key Information Extraction:")
    qa_pipeline = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")
    answer = qa_pipeline(question=question, context=text)
    print(f"Question: {question}\nAnswer: {answer['answer']}\n")


user_text_en = "to be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune."
translate_to_ukrainian(user_text_en)

image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7DUQxIxWkEtk2hEAVaWYMypIiwkr9TtUCIA&s"
classify_image(image_url)

prompt = "How to cook"
generate_text(prompt)

user_text_uk = "Будівництво будинку — це більше, ніж просто процес створення стін і даху. Це народження місця, де кожна деталь втілює мрії, де панує затишок, і куди хочеться повертатися знову й знову.Усе починається з фундаменту — не лише фізичного, а й морального. Це планування, розрахунки, підготовка до майбутнього. Фундамент символізує міцну основу для родини, стабільність і впевненість у завтрашньому дні.Створення стін і даху — це турбота про безпеку. Це те, що захищає від негоди й забезпечує спокій. Однак, дім не був би домом без тепла всередині — без людей, які його наповнюють радістю, сміхом і любов’ю.Кожна кімната має свій характер: кухня стає осередком сімейних зустрічей, спальня — місцем спокою, а вітальня — простором для радості й спілкування.Будинок — це також про турботу про майбутнє. Використання екологічно чистих матеріалів, продумане планування енергоефективності — це кроки до гармонії з природою й економії ресурсів.Побудова дому — це шлях, який поєднує фізичну працю з духовною творчістю. І головне, це про створення простору, де народжуються щастя, мрії та любов."
question_uk = "Як побудувати дім"
extract_information(user_text_uk, question_uk)

######################################################### 2
######################################################### 1

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = '/kaggle/input/text-for-lab4/ukr.txt'
num_samples = 10000
latent_dim = 612


def load_data(file_path, num_samples):
    input_texts = []
    target_texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines[:num_samples]:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            input_text, target_text = parts[:2]
            target_text = f"\t{target_text}\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
    return input_texts, target_texts


input_texts, target_texts = load_data(data_path, num_samples)

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_word_index = input_tokenizer.word_index
input_vocab_size = len(input_word_index) + 1

output_tokenizer = Tokenizer(char_level=True)
output_tokenizer.fit_on_texts(target_texts)
output_sequences = output_tokenizer.texts_to_sequences(target_texts)
output_word_index = output_tokenizer.word_index
output_vocab_size = len(output_word_index) + 1

max_encoder_seq_length = max([len(seq) for seq in input_sequences])
max_decoder_seq_length = max([len(seq) for seq in output_sequences])

encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(output_sequences, maxlen=max_decoder_seq_length, padding='post')

decoder_target_data = np.zeros((len(output_sequences), max_decoder_seq_length, output_vocab_size), dtype="float32")
for i, seq in enumerate(output_sequences):
    for t, word_id in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, word_id] = 1.

encoder_inputs = Input(shape=(None,), name="encoder_inputs")
encoder_embedding = Embedding(input_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,), name="decoder_inputs")
decoder_embedding = Embedding(output_vocab_size, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(output_vocab_size, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

batch_size = 64
epochs = 80
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

model.save("seq2seq_translation_model.h5")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Втрата тренування')
plt.plot(history.history['val_loss'], label='Втрата валідації')
plt.title('Втрата моделі')
plt.xlabel('Епохи')
plt.ylabel('Втрата')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Точність тренування')
plt.plot(history.history['val_accuracy'], label='Точність валідації')
plt.title('Точність моделі')
plt.xlabel('Епохи')
plt.ylabel('Точність')
plt.legend()

plt.show()

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_word_index.get('\t', 0)

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in output_word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word is None:
            break

        decoded_sentence += sampled_word

        if sampled_word == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence


while True:
    user_input = input("Введіть англійське речення (або 'exit' для виходу): ")
    if user_input.lower() == 'exit':
        print("Вихід з програми.")
        break

    input_seq = input_tokenizer.texts_to_sequences([user_input])
    input_seq = pad_sequences(input_seq, maxlen=max_encoder_seq_length, padding='post')
    translated_sentence = decode_sequence(input_seq)
    print(f"Переклад: {translated_sentence.strip()}")

######################################################### 1
######################################################### 3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

batch_size = 128
image_size = 32
latent_dim = 200
num_epochs = 30
learning_rate = 0.0002
beta1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.CIFAR100(root="./data", download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

generator = Generator(latent_dim).to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

for epoch in range(1, num_epochs + 1):
    for i, (images, _) in enumerate(dataloader):
        # Train Discriminator
        images = images.to(device)
        batch_size = images.size(0)
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch}/{num_epochs}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    grid = make_grid(fake_images, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Generated Images at Epoch {epoch}")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()

######################################################### 3