import torch
import pymongo
import yaml, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from llama_index.core.prompts import (
                                ChatMessage,
                                MessageRole,
                                ChatPromptTemplate,
                                )
from transformers import AutoTokenizer, AutoModel
from llama_index.llms.groq import Groq

with open('secrets.yaml') as f:
    secrets = yaml.load(f, Loader=yaml.FullLoader)

os.environ["GROQ_API_KEY"] = secrets['GROQ_API_KEY']
completion_llm = Groq(
                    model="llama3-70b-8192", 
                    api_key=os.environ["GROQ_API_KEY"],
                    temperature=0.0
                    )

model_answer = AutoModel.from_pretrained('models/answer_evaluation', trust_remote_code=True)
tokenizer_answer = AutoTokenizer.from_pretrained('models/answer_evaluation', trust_remote_code=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_answer.to(device)
model_answer.eval()

print("Answer Evaluation App Model Loaded Successfully !!!")

try:
    client = pymongo.MongoClient(os.environ["MONGO_DB_URI"])
    db = client['Elearning']
    qna_collection = db['qna']

except Exception as e:
    print(e)


def mean_pooling(
                model_output, 
                attention_mask
                ):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentence):
    encoded_input = tokenizer_answer([sentence], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model_answer(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def inference_answer_evaluation(
                                question,
                                answer01, 
                                answer02,
                                PRMT_TMPL = """
                                You have given Programming related question, correct answer and candidate answer. Rate the candidate answer on a scale of 0 to 5 based on how well it answers the question.

                                Question: {question}
                                Correct Answer: {correct_answer}
                                Candidate Answer: {candidate_answer}
                                Answer Rating:

                                Return only the rating score as a integer value.
                                """
                                ):
        try:
            sys_template = ChatPromptTemplate(
                                            message_templates=[
                                                            ChatMessage(
                                                                    role=MessageRole.SYSTEM, 
                                                                    content=PRMT_TMPL
                                                                    )
                                                            ]
                                            )
            fmt_messages = sys_template.format_messages(
                                                        question=question,
                                                        correct_answer=answer01,
                                                        candidate_answer=answer02
                                                        )
            chat_response = completion_llm.chat(fmt_messages)
            raw_output = int(chat_response.message.content)
            int_out = min(100, max(0, int(raw_output * 20) + np.random.randint(-10, 10)))

            int_out = round(int_out, 2)
            rating_score = f"{int_out} %"

        except:
            inf_encoded_input_01 = tokenizer_answer(
                                                answer01, 
                                                padding='max_length', 
                                                truncation=True, 
                                                max_length=50, 
                                                return_tensors='pt'
                                                )
            
            inf_encoded_input_02 = tokenizer_answer(
                                                answer02, 
                                                padding='max_length', 
                                                truncation=True, 
                                                max_length=50, 
                                                return_tensors='pt'
                                                )

            with torch.no_grad():
                    model_output_01 = model_answer(**inf_encoded_input_01.to(device))
                    model_output_02 = model_answer(**inf_encoded_input_02.to(device))

            sentence_embeddings_01 = mean_pooling(model_output_01, inf_encoded_input_01['attention_mask'])
            sentence_embeddings_01 = F.normalize(sentence_embeddings_01, p=2, dim=1)

            sentence_embeddings_02 = mean_pooling(model_output_02, inf_encoded_input_02['attention_mask'])
            sentence_embeddings_02 = F.normalize(sentence_embeddings_02, p=2, dim=1)
            cosine_score = F.cosine_similarity(x1=sentence_embeddings_01, x2=sentence_embeddings_02)
            rating_score = cosine_score * 100
            rating_score = rating_score.cpu().numpy().squeeze()
            rating_score = max(0, rating_score)
            rating_score = min(100, rating_score)
            rating_score = f"{round(float(rating_score), 2)} %" 

        response = {
                    "question": f"{question}",
                    "correct_answer": f"{answer01}",
                    "user_answer": f"{answer02}",
                    "rating_score": rating_score
                    }       
        
        qna_collection.insert_one(response)
        return rating_score