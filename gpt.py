from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
import requests
import openai

openai.api_key = MY_API_KEY

def query(input):
    response = openai.Completion.create(
        model="davinci-instruct-beta",
        prompt=input,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


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
    output = query(input = input)
    pred = True if "True" in output else False
    # print("text: "+ str(text))
    print("pred: "+ str(pred))
    outputs.append(pred)

eval_true = 0
for i in range(len(outputs)):
    if (outputs[i] == answers[i]):
        eval_true += 1
eval_acc = eval_true / len(outputs)
print("eval_accuracy: " + str(eval_acc))

