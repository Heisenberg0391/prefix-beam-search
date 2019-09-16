# coding=utf-8
from collections import defaultdict, Counter
from string import ascii_lowercase
import re
import numpy as np

def prefix_beam_search(ctc, lm=None, k=5, alpha=0.30, beta=5, prune=0.001):
    """
    Performs prefix beam search on the output of a CTC network.

    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        lm (func): Language model function. Should take as input a string and output a probability.
        k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.

    Retruns:
        string: The decoded CTC output.
    """

    lm = (lambda l: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    alphabet = list(ascii_lowercase) + [' ', '>', '%']
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):  # 遍历时刻
        # 去除每一步概率过小的类别以加速计算
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:  # 遍历前缀

            if len(l) > 0 and l[-1] == '>':  # 如果某个前缀整句结束则跳过
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)  # 剪枝后的字典
                # END: STEP 2

                # STEP 3: extend空白符
                if c == '%':  # 前缀不变，更新概率
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3
                else:
                    # STEP 4: extend与前缀结尾相同的非空字符
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        # 当非空字符与前缀结尾相同时只有非空概率Pnb，有两种情况
                        # (1)解码时没合并，要求前缀的路径以空白符结尾
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        # (2)解码时合并，要求前缀的路径以非空字符结尾
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    # END: STEP 4

                    # STEP 5: Extend任意非空字符并考虑语言模型
                    elif len(l.replace(' ', '')) > 0 and c in (' ', '>'):
                        # 如果该前缀遇到分词，则求当前前缀的语言模型得分
                        lm_prob = lm(l_plus.strip(' >')) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        # Extend任意非空字符，且还没分词则不需要考虑语言模型
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

        # STEP 7: 筛选并保留前beam width个前缀
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        if len(A_next)>1:
            pass
        # END: STEP 7

    return A_prev[0].strip('>')