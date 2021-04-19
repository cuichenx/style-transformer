import math
import os

from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch

# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)

def preprocess(txt):
    txt = txt.replace(" n't", "n't")
    txt = txt.replace("``", "\"")
    txt = txt.replace("''", "\"")
    return txt

def eval_ppl(fn, name):
    # fn = "/h/cuichenx/Downloads/StyleTransRes.txt"
    save_name = f"myppl_{name}.txt"
    if os.path.exists(save_name):
        with open(save_name) as f:
            scores = [float(s) for s in f.read().split()]

        scores.sort()
        scores = scores[:990]
    else:
        with open(fn, 'r') as f:
            a = [sent for sent in f.read().split('\n') if sent]
        assert len(a) == 1000
        print(fn)
        scores = [score(i) for i in a]
        scores = [s for s in scores if not math.isnan(s)]
        with open(f'myppl_{name}.txt', 'w') as f:
            f.writelines([str(s) + '\n' for s in scores])
    print(sum(scores) / len(scores))


# a=['there is a book on the desk , the book is nice',
#                 'there is a plane on the desk',
#                         'there is a book in the desk'
#    ]
a = ["it didn't taste watered down at all."]
scores = [score(i) for i in a]
print(scores)


eval_ppl("/h/cuichenx/Downloads/StyleTransRes.txt", 'st')
eval_ppl("/h/cuichenx/Downloads/outputext_step1_eps5.txt", 'ale')
eval_ppl("/h/cuichenx/Downloads/human.txt", 'human')