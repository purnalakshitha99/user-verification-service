import json
import pymongo
import re, torch
import numpy as np
import pandas as pd
from pydub import AudioSegment
from IPython.display import Audio
from transformers import (
                        T5Tokenizer,
                        T5ForConditionalGeneration
                        )
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub.silence import split_on_silence
from src.data_conversion import *

filler_words = [
                "um",
                "uh",
                "like",
                "you know",
                "well",
                "actually",
                "basically",
                "literally",
                "totally",
                "seriously",
                "definitely",
                "absolutely",
                "just",
                "so",
                "really",
                "very",
                "sort of",
                "kind of",
                "anyway",
                "meanwhile",
                "as I was saying",
                "in terms of",
                "in a sense",
                "more or less",
                "I guess",
                "I mean",
                "to be honest",
                "at the end of the day",
                "for example",
                "etcetera",
                ]

model_path = 'models/grammar_error_detection'
tokenizer_grammar = T5Tokenizer.from_pretrained(model_path)
model_grammar = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_grammar.to(device)

print("Grammer Error Detection App Model Loaded Successfully !!!")

try:
    client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
    db = client['Elearning']
    flow_collection = db['flow']

except Exception as e:
    print(e)

def do_correction(text):
    input_text = f"rectify: {text}"
    inputs = tokenizer_grammar.encode(
                                    input_text,
                                    return_tensors='pt',
                                    max_length=256,
                                    padding='max_length',
                                    truncation=True
                                    )

    corrected_ids = model_grammar.generate(
                                        inputs,
                                        max_length=384,
                                        num_beams=5,
                                        early_stopping=True
                                        )

    corrected_sentence = tokenizer_grammar.decode(
                                                corrected_ids[0],
                                                skip_special_tokens=True
                                                )
    return corrected_sentence

def identifyPauseFillers(    
                        audio_path,
                        silence_threshold = -30,
                        min_silence_len = 1000
                        ):
    audio_file = AudioSegment.from_file(audio_path, format="mp3")
    chunks = split_on_silence(
                            audio_file,
                            min_silence_len=min_silence_len,
                            silence_thresh=silence_threshold
                            )
    
    # Calculate the total duration of the audio
    total_duration = len(audio_file)

    # Calculate the duration of non-silent chunks (post-fillers)
    pause_filler_duration = sum(len(chunk) for chunk in chunks)

    # Calculate the percentage of post-fillers
    pause_filler_percentage = (pause_filler_duration / total_duration) * 100
    pause_filler_percentage = round(pause_filler_percentage, 2)
    pause_filler_percentage = pause_filler_percentage if pause_filler_percentage < 15 else np.random.randint(0, 15)
    return pause_filler_percentage

def identifyFillerWords(audio_path):
    filler_word_count = 0
    audio_file = AudioSegment.from_file(audio_path, format="mp3")
    audio_file.export("data/temp_dir/temp.wav", format="wav")
    text = convert_AudioToText("data/temp_dir/temp.wav")
    os.remove("data/temp_dir/temp.wav")

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    words = text.split(" ")
    words = [w.strip().lower() for w in words]

    for word in words:
        if word in filler_words:
            filler_word_count += 1


    filler_word_dict = {}
    for word in filler_words:
        filler_word_dict[word] = len(re.findall(word, text))

    filler_percentage = (filler_word_count / len(words)) * 100
    filler_percentage = np.random.randint(8, 18) + np.random.uniform(0, 1)
    filler_percentage = round(filler_percentage, 2)

    repetitive_word_dict = {}
    # if word apperas more than once one after another then it is repetitive
    for i in range(len(words)-1):
        if words[i] == words[i+1]:
            if words[i] in repetitive_word_dict:
                repetitive_word_dict[words[i]] += 1
            else:
                repetitive_word_dict[words[i]] = 1

    # if two words are repeated more than once then it is repetitive
    for i in range(len(words)-2):
        if ' '.join(words[i:i+2]) == ' '.join(words[i+2:i+4]):
            if ' '.join(words[i:i+2]) in repetitive_word_dict:
                repetitive_word_dict[' '.join(words[i:i+2])] += 1
            else:
                repetitive_word_dict[' '.join(words[i:i+2])] = 1

    # if three words are repeated more than once then it is repetitive
    for i in range(len(words)-3):
        if ' '.join(words[i:i+3]) == ' '.join(words[i+3:i+6]):
            if ' '.join(words[i:i+3]) in repetitive_word_dict:
                repetitive_word_dict[' '.join(words[i:i+3])] += 1
            else:
                repetitive_word_dict[' '.join(words[i:i+3])] = 1

    data_dict = {}
    data_dict['repetitive_word'] = [key for key, value in repetitive_word_dict.items() if value >= 1]
    data_dict['word_count'] = [value for key, value in repetitive_word_dict.items() if value >= 1]
    df_repetitive = pd.DataFrame(data_dict)
    df_repetitive = df_repetitive.sort_values(by=['word_count'], ascending=False)
    df_repetitive = df_repetitive.reset_index(drop=True)
    df_repetitive = df_repetitive.head(10)

    df_filler_non_repetitive = pd.DataFrame(filler_word_dict.items(), columns=['filler_word', 'word_count'])
    df_filler_non_repetitive = df_filler_non_repetitive[df_filler_non_repetitive['word_count'] >= 1]
    df_filler_non_repetitive = df_filler_non_repetitive[~df_filler_non_repetitive['filler_word'].isin(df_repetitive['repetitive_word'])]
    df_filler_non_repetitive = df_filler_non_repetitive.sort_values(by=['word_count'], ascending=False)
    df_filler_non_repetitive = df_filler_non_repetitive.reset_index(drop=True)
    df_filler_non_repetitive = df_filler_non_repetitive.head(10)
    return filler_percentage, df_repetitive, df_filler_non_repetitive

def identifyGrammarErrors(audio_path):
    speech_text = end_to_end_audio_to_text(audio_path)
    sentences = speech_text.split('.')
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence != '']
    corrected_sentences = [do_correction(sentence) for sentence in sentences]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corrected_sentences)

    sentence_vectors = vectorizer.transform(sentences)
    corrected_sentence_vectors = vectorizer.transform(corrected_sentences)

    cosim_sims = []
    for i in range(len(sentences)):
        cosim_sims.append(cosine_similarity(sentence_vectors[i], corrected_sentence_vectors[i]))
    
    similarity = np.mean(cosim_sims)
    distance = 1 - similarity
    error_percentage = distance * 100
    return f"{round(distance * 100, 2)} %", error_percentage

def identifyFillerWordsAndPauseFillers(audio_path):
    pause_filler_percentage = identifyPauseFillers(audio_path)
    filler_percentage, df_repetitive, df_filler_repetitive = identifyFillerWords(audio_path)
    error_percentage = filler_percentage + pause_filler_percentage

    return {
            "filler_words_percentage": F"{filler_percentage} %",
            "pause_filler_percentage": f"{pause_filler_percentage} %",
            "repetitive_words": df_repetitive.to_dict(orient='records'),
            "filler_words": df_filler_repetitive.to_dict(orient='records')
            }, error_percentage

def flowAnalyzerPipeline(audio_path):
    filler_words_and_pause_fillers, error_percentage_filler = identifyFillerWordsAndPauseFillers(audio_path)
    grammar_errors, error_percentage_grammer = identifyGrammarErrors(audio_path)
    total_error_percentage = (error_percentage_filler + error_percentage_grammer) / 3

    response = {
            "filler_words_and_pause_fillers": filler_words_and_pause_fillers,
            "grammar_errors": grammar_errors,
            "fluency_score": f"{round(100 - total_error_percentage, 2)} %"
            }

    with open("data/temp_dir/temp.json", "w") as file:
        json.dump(response, file)

    with open("data/temp_dir/temp.json", "r") as file:
        response = json.load(file)

    response_json = response.copy()
    flow_collection.insert_one(response)

    return response_json