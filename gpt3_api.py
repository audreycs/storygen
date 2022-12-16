import openai
from transformers import GPT2TokenizerFast
from memory import Memory

openai.api_key = "27710ad6c85b412d99bd9d0f92d4b246"
openai.api_base =  "https://sg-001.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2022-06-01-preview' # this may change in the future
deployment_id='sandbox-dv2' #This will correspond to the custom name you chose for your deployment when you deployed a model.

# openai.Model.retrieve("text-curie-001")

# Send a completion call to generate an answer
def gpt3_k2s(logger, kwlist, prompt_model):
    s = ', '.join(kwlist)
    start_phrase = "Generate a sentence containing "+s+":"

    response = openai.Completion.create(model=prompt_model,
                                        engine=deployment_id, 
                                        prompt=start_phrase, 
                                        max_tokens=100
                                        )
    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    return text

def gpt3_generation(logger, kw_list, prompt, scores, stem_to_words, beta_, freq_pen, story_model, hub_word_bias):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    real_key_value = dict()
    for k, v in scores.items():
        real_words = stem_to_words[k]
        for rw in real_words:
            real_key_value[rw] = v
    
    keys = list(real_key_value.keys())
    values = [0]*len(keys)

    mem = Memory(kw_list)
    mem.history_update(prompt)

    while mem.not_end():
        current_kw = mem.kw_list[mem.process]
        mem.update()

        for idx, k in enumerate(keys):
            if k in current_kw:
                values[idx] = hub_word_bias  # extremely large bias for hub words
            else:
                values[idx] = real_key_value[k]*beta_

        logger.info(f"{dict(zip(keys, values))}")

        _keys = [' '+k for k in keys]
        token_ids = [tokenizer(k)['input_ids'][0] for k in _keys]
        token_score_dict = dict(zip(token_ids, values))

        start_phrase = 'Continue the story beginning with \"' + mem.history + "\""
        response = openai.Completion.create(model=story_model,
                                            engine=deployment_id, 
                                            prompt=start_phrase, 
                                            max_tokens=200,
                                            temperature=1,
                                            top_p=1,
                                            frequency_penalty=freq_pen,
                                            logit_bias = token_score_dict
                                            )
        
        text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()

        mem.history_update("\n\n" + text)
        # if prompt in text:
        #     continuation = text
        # else:
        #     continuation = prompt+' '+text
    
    return mem.history

def original_gpt3(logger, kw_list, story_model):
    kw_list = ["\""+w+"\"" for w in kw_list]
    kw_string = ', '.join(kw_list)
    start_phrase = "Generate a story using plots: "+kw_string

    # logger.info(start_phrase)
    # if args.model == 'ada':
    #     gpt3_model = 'text-ada-001'
    # elif args.model == 'babbage':
    #     gpt3_model = 'text-babbage-001'
    # elif args.model == 'curie':
    #     gpt3_model = 'text-curie-001'
    # elif args.model == 'davinci':
    #     gpt3_model = 'text-davinci-002'
    # else:
    #     logger.info("Using generation model \"text-curie-001\" for story generation!")
    #     gpt3_model = 'text-curie-001'

    response = openai.Completion.create(model=story_model,
                                        engine=deployment_id,
                                        prompt=start_phrase,
                                        max_tokens=500,
                                        temperature=1,
                                        top_p=1,   # top_p与temperature只改变一个
                                        n=1,
                                        presence_penalty=0,
                                        frequency_penalty=0,
                                        logprobs=None,
                                        )

    # for choice in response['choices']:
    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    # with open('test.txt', 'w', encoding='utf-8') as f:
    #     f.write(text)
    #     f.write('\n')
    return text


def customed_gpt3():
    s = '''
        Please use PyTorch to write a Knowledge Graph Embedding Model. Remember to present the code in good format.
        '''
    response = openai.Completion.create(model='text-davinci-003',
                                        engine=deployment_id,
                                        prompt=s,
                                        max_tokens=1000,
                                        temperature=1,
                                        top_p=1,   # top_p与temperature只改变一个
                                        )

    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    # print(f"davinci-003: {text}")

customed_gpt3()