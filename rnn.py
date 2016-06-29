## Python Minimal Character-based Recurrent Neural Network
## http://karpathy.github.io/2015/05/21/rnn-effectiveness/

############################
########## CONFIG ##########
############################

snapshot_every = 2000
sample_every = 250
sample_size = 1024
input_filepath = 'input.txt'
hidden_size = 125
learning_rate = 0.1
learning_rate_decay = 0.97
learning_rate_decay_after = 10
seq_length = 20
seed = 1234
temperature = 1.0 # 0.75-1.25

############################
########## CONFIG ##########
############################

title = 'Minimal Character-based Recurrent Neural Network'
version = '2.0.1'

import sys, os.path
import numpy as np
from time import gmtime, strftime

def softmax(w, temp=1.0):
    e = np.exp(np.array(w)/temp)
    dist = e / np.sum(e)
    return dist

def softmax_1(x): # temp=1.0
    e_x = np.exp(x - np.max(x))
    out = e_x / np.sum(e_x)
    return out

def passes(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        # update the hidden state
        hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + bh)
        # compute the output vector (unnormalized log prob'ys for next chars)
        ys[t] = np.dot(W_hy, hs[t]) + by
        ps[t] = softmax_1(ys[t]) # prob'ys for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    dW_xh = np.zeros_like(W_xh)
    dW_hh, dW_hy = np.zeros_like(W_hh), np.zeros_like(W_hy)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y
        dW_hy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(W_hy.T, dy) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dW_xh += np.dot(dhraw, xs[t].T)
        dW_hh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(W_hh.T, dhraw)
    for dparam in [dW_xh, dW_hh, dW_hy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # mitigate exploding gradients
    return loss, dW_xh, dW_hh, dW_hy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    # this function returns a sample of integers from the network
    # h = memory state, seed_ix = seed_letter
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + bh)
        y = np.dot(W_hy, h) + by
        p = softmax(y, temperature)
        # Next letter is a random one weighted by the probabilities calculated
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

# timestamp for console output
def GetTimestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

if __name__ == '__main__':

    print('[%s] Version: %s | %s' % (GetTimestamp(), version, title))
    print('[%s] Initializing...' % (GetTimestamp()), end='', flush=True)

    output_filepath = 'output.txt'
    snapshot_filepath = 'nndata.npz'

    input_data = open(input_filepath, 'r').read()
    chars = list(set(input_data))
    data_size, vocab_size = len(input_data), len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    if seed > 0:
      np.random.seed(seed)
    W_xh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
    W_hh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    W_hy = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
    bh = np.zeros((hidden_size, 1)) # hidden bias
    by = np.zeros((vocab_size, 1)) # output bias

    n, p = 0, 0
    epoch = 0;
    mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/vocab_size) * seq_length # loss at iteration 0

    if os.path.isfile(snapshot_filepath):
        loaded = np.load(snapshot_filepath)
        hidden_size, seq_length = loaded['hidden_size'], loaded['seq_length']
        W_xh, W_hh, W_hy = loaded['W_xh'], loaded['W_hh'], loaded['W_hy']
        bh, by = loaded['bh'], loaded['by']
        n, p = loaded['n'], loaded['p']
        epoch = loaded['epoch']
        mW_xh, mW_hh, mW_hy = loaded['mW_xh'], loaded['mW_hh'], loaded['mW_hy']
        mbh, mby = loaded['mbh'], loaded['mby']
        smooth_loss, hprev = loaded['smooth_loss'], loaded['hprev']
        if epoch > learning_rate_decay_after:
            learning_rate *= pow(learning_rate_decay, epoch - learning_rate_decay_after)

    print('ok')
    print('[%s] Data info | Total size: %i | Unique characters: %i' % (GetTimestamp(), data_size, vocab_size))
    print('[%s] epoch: %i | n: %i | hidden_size: %i | seq_length: %i | learning_rate: %s | learning_rate_decay: %s | seed: %i | temperature: %s'
          % (GetTimestamp(), epoch, n, hidden_size, seq_length, str(learning_rate), str(learning_rate_decay), seed, str(temperature)))

    while True:
        if p + seq_length + 1 >= len(input_data) or n == 0: 
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            epoch += 1
            p = 0

        if epoch > learning_rate_decay_after:
            learning_rate *= pow(learning_rate_decay, epoch - learning_rate_decay_after)
            if learning_rate > 0.05:
                print('[%s] Reached epoch %i | learning_rate: %i' % (GetTimestamp(), learning_rate))

        inputs, targets = [char_to_ix[ch] for ch in input_data[p:p + seq_length]], [char_to_ix[ch] for ch in input_data[p+1:p + seq_length+1]]

        if n > 0 and n % sample_every == 0:
            sample_ix = sample(hprev, inputs[0], sample_size)
            text = ''.join(ix_to_char[ix] for ix in sample_ix)
            with open(output_filepath, 'a') as fp:
                fp.write(text + '\n')
            print('[%s] Saved sample. | Filepath: %s | epoch: %i | Iteration: %i | loss: %s | temperature: %s'
                  % (GetTimestamp(), output_filepath, epoch, n, str(smooth_loss), str(temperature)))

        # forward seq_length characters through the net and fetch gradient
        loss, dW_xh, dW_hh, dW_hy, dbh, dby, hprev = passes(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([W_xh, W_hh, W_hy, bh, by],
                                      [dW_xh, dW_hh, dW_hy, dbh, dby],
                                      [mW_xh, mW_hh, mW_hy, mbh, mby]):
            mem += dparam**2
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
      
        p += seq_length # move data pointer
        n += 1 # iteration counter 

        if n > 0 and n % snapshot_every == 0:
            np.savez_compressed(snapshot_filepath, hidden_size=hidden_size, seq_length=seq_length, W_xh=W_xh, W_hh=W_hh, W_hy=W_hy, bh=bh, by=by,
                n=n, p=p, epoch=epoch, mW_xh=mW_xh, mW_hh=mW_hh, mW_hy=mW_hy, mbh=mbh, mby=mby, smooth_loss=smooth_loss, hprev=hprev)
            print('[%s] Saved snapshot. | Filepath: %s | epoch: %d | n: %d | W_hh: %dx%d' % (GetTimestamp(), snapshot_filepath, epoch, n, len(W_hh), len(W_hh[0])))
