"""
Method that takes learned cross-lingual embeddings as input and produces word-to-word translations.
"""

import argparse
import torch
import torch.nn.functional as F
import sys
import os

from src.models import build_model
from src.utils import get_nn_avg_dist


def get_word_translations(emb1, emb2, knn, softmax_temp=30.):
    """
    Given source and target word embeddings, and a list of source words,
    produce a list of lists of k-best translations for each source word.
    """
    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # we always use the contextual dissimilarity measure as this gives the best performance (csls_knn_10)
    # calculate the average distances to k nearest neighbors
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)

    top_k_match_ids = []
    step_size = 1000

    for i in range(0, emb1.shape[0], step_size):
        print('Processing word ids %d-%d...' % (i, i+step_size))
        word_ids = range(i, i+step_size)

        # use the embeddings of the current word ids
        query = emb1[word_ids]

        # calculate the scores with the contextual dissimilarity measure
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[word_ids][:, None] + average_dist2[None, :])

        # get the indices of the highest scoring target words
        top_sim_scores, top_match_ids = scores.topk(knn, 1, True)  # returns a (values, indices) tuple (same as torch.topk)
        top_sim_scores = F.softmax(softmax_temp * top_sim_scores, 1)
        top_k_match_ids += [(ids, scores) for ids, scores in zip(top_match_ids, top_sim_scores)]

    return top_k_match_ids


def main(args):
    assert os.path.exists(args.src_emb)
    assert os.path.exists(args.tgt_emb)
    src_emb, tgt_emb, mapping, _ = build_model(args, False)

    # get the mapped word embeddings as vectors of shape [max_vocab_size, embedding_size]
    src_emb = mapping(src_emb.weight).data
    tgt_emb = tgt_emb.weight.data

    id2word1 = {id_: word for word, id_ in args.src_dico.word2id.items()}
    id2word2 = {id_: word for word, id_ in args.tgt_dico.word2id.items()}

    top_k_match_ids = get_word_translations(src_emb, tgt_emb, args.knn)

    output_file = '%s-%s.txt' % (args.src_lang, args.tgt_lang)
    print('Writing to %s...' % output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for src_id, (tgt_ids, tgt_scores) in enumerate(top_k_match_ids):
            for tgt_id, score in zip(tgt_ids, tgt_scores):
                if args.cuda:
                    tgt_id, score = tgt_id.cpu(), score.cpu()
                if args.output_scores:
                    f.write('%s %s %.4f\n' % (id2word1[src_id],
                                              id2word2[int(tgt_id.numpy())],
                                              float(score.numpy())))
                else:
                    f.write('%s %s\n' % (id2word1[src_id],
                                         id2word2[int(tgt_id.numpy())]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src-lang', required=True, help='Source language')
    parser.add_argument('--tgt-lang', required=True, help='Target language')
    parser.add_argument('-s', '--src-emb', required=True, help='The path to the source language embeddings')
    parser.add_argument('-t', '--tgt-emb', required=True, help='The path to the target language embeddings')
    parser.add_argument('--max-vocab', type=int, default=500000, help='Maximum vocabulary size')
    parser.add_argument('--emb-dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--cuda', action='store_true', help='Run on GPU')
    parser.add_argument("--normalize_embeddings", type=str, default='', help="Normalize embeddings before training")
    parser.add_argument("--output_scores", dest='output_scores', action='store_true',
                        help="Whether normalized scores should be included in output")
    parser.set_defaults(output_scores=False)
    parser.add_argument('--softmax_temp', type=float, default=1./30.,
                        help='Softmax temperature for score normalization')

    parser.add_argument('--knn', type=int, default=10,
                        help='K-NNs that should be retrieved for each source word'
                             '(Conneau et al. use 10 for evaluation)')
    args = parser.parse_args()
    if not args.cuda:
        print('Not running on GPU...', file=sys.stderr)

    main(args)
