import os
import requests
import json
import openai

openai.api_key = "27710ad6c85b412d99bd9d0f92d4b246"
openai.api_base =  "https://sg-001.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2022-06-01-preview' # this may change in the future

deployment_id='sandbox-dv2' #This will correspond to the custom name you chose for your deployment when you deployed a model.

# openai.Model.retrieve("text-curie-001")

# Send a completion call to generate an answer
def gpt3_k2s(kwlist):
    s = ', '.join(kwlist)
    start_phrase = "Generate a sentence containing "+s+":"

    response = openai.Completion.create(model='text-curie-001',
                                        engine=deployment_id, 
                                        prompt=start_phrase, 
                                        max_tokens=100
                                        )
    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    return text

def gpt3_test():
    start_phrase = """
                   One day I saw a couple on the street.
                   """

    response = openai.Completion.create(model='text-curie-001',
                                        engine=deployment_id,
                                        prompt=start_phrase,
                                        max_tokens=100,
                                        temperature=1,
                                        top_p=1,   # top_p与temperature只改变一个
                                        n=1,
                                        presence_penalty=0,
                                        frequency_penalty=0,
                                        logprobs=None,
                                        )

    for choice in response['choices']:
        text = choice['text'].replace('\n', '').replace(' .', '.').strip()
        with open('test.txt', 'w', encoding='utf-8') as f:
            f.write(text)
            f.write('\n')
        # logprob = choice['logprobs']
        # print(logprob)

gpt3_test()