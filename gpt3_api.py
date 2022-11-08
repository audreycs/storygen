import openai
from transformers import GPT2TokenizerFast

openai.api_key = "27710ad6c85b412d99bd9d0f92d4b246"
openai.api_base =  "https://sg-001.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2022-06-01-preview' # this may change in the future
deployment_id='sandbox-dv2' #This will correspond to the custom name you chose for your deployment when you deployed a model.

# openai.Model.retrieve("text-curie-001")

# Send a completion call to generate an answer
def gpt3_k2s(logger, args, kwlist):
    s = ', '.join(kwlist)
    start_phrase = "Generate a sentence containing "+s+":"

    if args.prompt_model == 'ada':
        gpt3_model = 'text-ada-001'
    elif args.prompt_model == 'babbage':
        gpt3_model = 'text-babbage-001'
    elif args.prompt_model == 'curie':
        gpt3_model = 'text-curie-001'
    elif args.prompt_model == 'davinci':
        gpt3_model = 'text-davinci-002'
    else:
        logger.info("Using generation model \"text-davinci-001\" for story generation!")
        gpt3_model = 'text-davinci-001'

    response = openai.Completion.create(model=gpt3_model,
                                        engine=deployment_id, 
                                        prompt=start_phrase, 
                                        max_tokens=100
                                        )
    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    return text

def gpt3_generation(logger, args, prompt, scores, stem_to_words):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    keys = list(scores.keys())
    values = [scores[k] for k in keys]
    _keys = [' '+k for k in keys]
    token_ids = [tokenizer(k)['input_ids'][0] for k in _keys]
    token_score_dict = dict(zip(token_ids, values))
    print(token_score_dict)

    if args.model == 'ada':
        gpt3_model = 'text-ada-001'
    elif args.model == 'babbage':
        gpt3_model = 'text-babbage-001'
    elif args.model == 'curie':
        gpt3_model = 'text-curie-001'
    elif args.model == 'davinci':
        gpt3_model = 'text-davinci-002'
    else:
        logger.info("Using generation model \"text-davinci-001\" for story generation!")
        gpt3_model = 'text-davinci-001'

    start_phrase = 'Continue the story beginning with \"' + prompt + "\""
    response = openai.Completion.create(model=gpt3_model,
                                        engine=deployment_id, 
                                        prompt=start_phrase, 
                                        max_tokens=500,
                                        frequency_penalty=0.1,
                                        logit_bias = token_score_dict
                                        )
    
    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    return text

def original_gpt3(logger, args, kw_list):
    kw_list = ["\""+w+"\"" for w in kw_list]
    kw_string = ', '.join(kw_list)
    start_phrase = "Generate a story containing plots: "+kw_string

    # logger.info(start_phrase)
    if args.model == 'ada':
        gpt3_model = 'text-ada-001'
    elif args.model == 'babbage':
        gpt3_model = 'text-babbage-001'
    elif args.model == 'curie':
        gpt3_model = 'text-curie-001'
    elif args.model == 'davinci':
        gpt3_model = 'text-davinci-002'
    else:
        logger.info("Using generation model \"text-davinci-001\" for story generation!")
        gpt3_model = 'text-davinci-001'

    response = openai.Completion.create(model=gpt3_model,
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