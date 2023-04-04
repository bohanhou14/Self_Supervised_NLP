from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
import requests

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
API_TOKEN = MY_API_TOKEN
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]

boolQ = load_dataset("boolq")

indices = torch.randperm(boolQ['train'].num_rows)

N_TRUE = 4
N_FALSE = 4
true_data = []
false_data = []
for idx in indices:
    id = idx.item()
    data = boolQ['train'][id]
    answer = data['answer']
    if (answer == True and N_TRUE > 0):
        N_TRUE -= 1
        true_data.append(data)
    elif (answer == False and N_FALSE > 0):
        N_FALSE -= 1
        false_data.append(data)
    elif (N_TRUE == 0 and N_FALSE == 0):
        break
    
few_shot_learning_prompt = ""
while(len(true_data) > 0 or len(false_data) > 0):
    if (len(true_data) < len(false_data)):
        data = false_data.pop()
    elif (len(true_data) == len(false_data)):
        data = true_data.pop()
    prompt = ""
    prompt += "question: " + data['question'] + "?\n"
    prompt += "answer: " + str(data['answer']) + ".\n"
    few_shot_learning_prompt += prompt
    few_shot_learning_prompt += "\n"


eval_indices = torch.randperm(boolQ['validation'].num_rows)
eval_indices = eval_indices[:10]
inputs = []
answers = []
for index in eval_indices:
    idx = index.item()
    data = boolQ['validation'][idx]
    input = few_shot_learning_prompt
    input += "question: " + data['question'] + "?\n"
    input += "answer: "
    inputs.append(input)
    # print(input)
    answers.append(bool(data['answer']))

outputs = []
for input in inputs:
    output = query(
    {
        "inputs": input
    })
    text = output['generated_text']
    text = text.split(' ')
    pred = True if text[-1] == 'True' else False
    # print("text: "+ str(text))
    print("pred: "+ str(pred))
    outputs.append(pred)

eval_true = 0
for i in range(len(outputs)):
    if (outputs[i] == answers[i]):
        eval_true += 1
eval_acc = eval_true / len(outputs)
print("eval_accuracy: " + str(eval_acc))

