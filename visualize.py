import os
import time

from data import load_dataset
from main import Config
from models import StyleTransformer
from train import get_lengths
from utils import tensor2text
import torch
import numpy as np


def calc_temperature(temperature_config, step):
    num = len(temperature_config)
    for i in range(num):
        t_a, s_a = temperature_config[i]
        if i == num - 1:
            return t_a
        t_b, s_b = temperature_config[i + 1]
        if s_a <= step < s_b:
            k = (step - s_a) / (s_b - s_a)
            temperature = (1 - k) * t_a + k * t_b
            return temperature

def my_eval(config, vocab, model_F, test_iters, global_step, temperature):
    model_F.eval()
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles

            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )

            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )

            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output

    pos_iter = test_iters.pos_iter
    neg_iter = test_iters.neg_iter

    gold_text, raw_output, rev_output = zip(inference(neg_iter, 0), inference(pos_iter, 1))

    yelp_ref0_path = 'evaluator/yelp.refs.0'
    yelp_ref1_path = 'evaluator/yelp.refs.1'
    ref_text = []
    with open(yelp_ref0_path, 'r') as fin:
        ref_text.append(fin.readlines())
    with open(yelp_ref1_path, 'r') as fin:
        ref_text.append(fin.readlines())

    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print('*' * 20, 'neg sample', '*' * 20)
        print('[gold]', gold_text[0][idx])
        print('[raw ]', raw_output[0][idx])
        print('[rev ]', rev_output[0][idx])
        print('[ref ]', ref_text[0][idx])

    print('*' * 20, '********', '*' * 20)

    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print('*' * 20, 'pos sample', '*' * 20)
        print('[gold]', gold_text[1][idx])
        print('[raw ]', raw_output[1][idx])
        print('[rev ]', rev_output[1][idx])
        print('[ref ]', ref_text[1][idx])

    print('*' * 20, '********', '*' * 20)

    # save output
    save_file = config.save_folder + '/' + str(global_step) + '.txt'

    with open(save_file, 'w') as fw:

        for idx in range(len(rev_output[0])):
            print('*' * 20, 'neg sample', '*' * 20, file=fw)
            print('[gold]', gold_text[0][idx], file=fw)
            print('[raw ]', raw_output[0][idx], file=fw)
            print('[rev ]', rev_output[0][idx], file=fw)
            print('[ref ]', ref_text[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

        for idx in range(len(rev_output[1])):
            print('*' * 20, 'pos sample', '*' * 20, file=fw)
            print('[gold]', gold_text[1][idx], file=fw)
            print('[raw ]', raw_output[1][idx], file=fw)
            print('[rev ]', rev_output[1][idx], file=fw)
            print('[ref ]', ref_text[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

    model_F.train()


def my_eval_interp(config, vocab, model_F, test_iters, global_step, temperature):
    model_F.eval()
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']
    ratios = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_outputs = [[] for _ in ratios]
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles

            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())

            for i, r in enumerate(ratios):
                with torch.no_grad():
                    rev_log_probs = model_F(
                        inp_tokens,
                        None,
                        inp_lengths,
                        raw_styles + (rev_styles-raw_styles)*r,  # 0.2, 0.4, 0.6, 0.8, 1.0
                        generate=True,
                        differentiable_decode=False,
                        temperature=temperature,
                    )

                rev_outputs[i] += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_outputs

    pos_iter = test_iters.pos_iter
    neg_iter = test_iters.neg_iter

    gold_text, raw_output, rev_outputs = zip(inference(neg_iter, 0), inference(pos_iter, 1))

    yelp_ref0_path = 'evaluator/yelp.refs.0'
    yelp_ref1_path = 'evaluator/yelp.refs.1'
    ref_text = []
    with open(yelp_ref0_path, 'r') as fin:
        ref_text.append(fin.readlines())
    with open(yelp_ref1_path, 'r') as fin:
        ref_text.append(fin.readlines())

    # for k in range(5):
    #     idx = np.random.randint(len(rev_output[0]))
    #     print('*' * 20, 'neg sample', '*' * 20)
    #     print('[gold]', gold_text[0][idx])
    #     print('[raw ]', raw_output[0][idx])
    #     print('[rev ]', rev_output[0][idx])
    #     print('[ref ]', ref_text[0][idx])
    #
    # print('*' * 20, '********', '*' * 20)
    #
    # for k in range(5):
    #     idx = np.random.randint(len(rev_output[1]))
    #     print('*' * 20, 'pos sample', '*' * 20)
    #     print('[gold]', gold_text[1][idx])
    #     print('[raw ]', raw_output[1][idx])
    #     print('[rev ]', rev_output[1][idx])
    #     print('[ref ]', ref_text[1][idx])
    #
    # print('*' * 20, '********', '*' * 20)

    # save output
    save_file = config.save_folder + '/' + str(global_step) + 'interp.txt'

    with open(save_file, 'w') as fw:

        for idx in range(len(rev_outputs[0][0])):
            print('*' * 20, 'neg sample', '*' * 20, file=fw)
            print('[gold]', gold_text[0][idx], file=fw)
            print('[raw  0.0]', raw_output[0][idx], file=fw)
            for i, r in enumerate(ratios):
                print(f'[rev {r: .1f}]', rev_outputs[0][i][idx], file=fw)
            print('[ref ]', ref_text[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

        for idx in range(len(rev_outputs[1][0])):
            print('*' * 20, 'pos sample', '*' * 20, file=fw)
            print('[gold]', gold_text[1][idx], file=fw)
            print('[raw  1.0]', raw_output[1][idx], file=fw)
            for i, r in enumerate(ratios):
                print(f'[rev {1-r: .1f}]', rev_outputs[1][i][idx], file=fw)
            print('[ref ]', ref_text[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

    model_F.train()



if __name__ == '__main__':
    config = Config()
    config.save_folder = config.save_path + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    os.makedirs(config.save_folder)
    os.makedirs(config.save_folder + '/ckpts')
    print('Save Path:', config.save_folder)

    train_iters, dev_iters, test_iters, vocab = load_dataset(config)

    # print(len(vocab))
    # for batch in test_iters:
    #     text = tensor2text(vocab, batch[0])
    #     print('\n'.join(text))
    #     print(batch.label)
    #     break

    model_F = StyleTransformer(config, vocab).to(config.device)
    global_step = 1200
    save_path = f"save/Mar27144631/{global_step}_F.pth"
    state_dict = torch.load(save_path)
    model_F.load_state_dict(state_dict)
    temperature = calc_temperature(config.temperature_config, global_step)

    my_eval_interp(config, vocab, model_F, test_iters, global_step, temperature)
