import os
import time

from matplotlib import pylab
from sklearn.manifold import TSNE

from data import load_dataset
from main import Config
from models import StyleTransformer
from train import get_lengths
from utils import tensor2text
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine as cosine_dist

class EmbeddingsNN:
    def __init__(self, vocab):
        self.vocab = vocab


    def cluster_embeddings(self, token_embed, style_embed):

        # normalize embeddings
        # token_embed /= np.linalg.norm(token_embed, ord=2, axis=1, keepdims=True)
        # style_embed /= np.linalg.norm(style_embed, ord=2, axis=1, keepdims=True)
        nbrs = NearestNeighbors(n_neighbors=100, metric='euclidean')
        self.all_embed = np.concatenate((token_embed, style_embed), axis=0)

        nbrs.fit(self.all_embed)
        self.nbrs = nbrs


    def closest_to(self, word, k=10):
        if word == 'NEG': i = len(vocab)
        elif word == 'POS': i = len(vocab)+1
        else: i = self.vocab[word]

        distances, indices = self.nbrs.kneighbors(self.all_embed[i:i+1], n_neighbors=k)
        print('Nearest Neighbours to', word)
        for i, d in zip(indices[0][1:], distances[0][1:]):
            if i == len(vocab): w = 'NEG'
            elif i == len(vocab) + 1: w = 'POS'
            else: w = vocab.itos[i]
            print(f'{w.ljust(20)}\t{d:.3f}')


    def tsne_plot_representation(self, limit=200):
        """Plot a 2-D visualization of the learned representations using t-SNE."""
        print('doing tsne...')
        mapped_X = TSNE(n_components=2).fit_transform(np.concatenate((self.all_embed[:limit], self.all_embed[-2:])))
        print('done')
        pylab.figure(figsize=(6, 8))

        for i, w in enumerate(self.vocab.itos[:limit]):
            pylab.text(mapped_X[i, 0], mapped_X[i, 1], w, color='k')

        pylab.text(mapped_X[-2, 0], mapped_X[-2, 1], 'NEG', color='r')
        pylab.text(mapped_X[-1, 0], mapped_X[-1, 1], 'POS', color='r')

        pylab.xlim(mapped_X[:, 0].min(), mapped_X[:, 0].max())
        pylab.ylim(mapped_X[:, 1].min(), mapped_X[:, 1].max())
        pylab.show()

    @staticmethod
    def stoi(word):
        if word == 'NEG': return len(vocab)
        if word == 'POS': return len(vocab)+1
        else:
            i = vocab[word]
            assert i>0, f"{word} is not in vocab"


    def cosine_sim(self, w1, w2):
        return 1 - cosine_dist(self.all_embed[self.stoi(w1)],
                               self.all_embed[self.stoi(w2)])


if __name__ == '__main__':
    config = Config()
    config.save_folder = config.save_path + '/' + str(time.strftime('%b%d%H%M%S', time.localtime()))
    os.makedirs(config.save_folder)
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

    token_embed = model_F.embed.token_embed.weight.detach().cpu().numpy()
    style_embed = model_F.style_embed.weight.detach().cpu().numpy()

    NN =  EmbeddingsNN(vocab)
    NN.cluster_embeddings(token_embed, style_embed)
    NN.tsne_plot_representation()
