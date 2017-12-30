import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from optparse import OptionParser
from tools.config_tools import Config

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/train_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)

USE_CUDA = opt.cuda


class Encoder(nn.Module):
    def __init__(self, num_layer=2, hidden_size=128):
        super(Encoder, self).__init__()

        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size=1024, hidden_size=hidden_size, num_layers=self.num_layer, batch_first=True,
                              bidirectional=False, dropout=0.1)

    def init_hidden(self, batch_size):
        bz = batch_size
        hidden_0 = Variable(torch.nn.init.orthogonal(torch.zeros(self.num_layer, bz, self.hidden_size)))
        if USE_CUDA:
            hidden_0 = hidden_0.cuda()
        return hidden_0

    def forward(self, vfeat, hidden):
        output, hidden = self.encoder(vfeat, hidden)

        return output, hidden


class Attn(nn.Module):
    def __init__(self, method='general', hidden_size=128):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        bs = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(bs, seq_len))  # B x 1 x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[:, i] = self.score(hidden, encoder_outputs[:, i, :])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden * energy
            energy = torch.sum(energy, dim=1)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoder(nn.Module):
    def __init__(self, attn_model='general', hidden_size=128, output_size=opt.afeat_pca, n_layers=2, dropout_p=0.1):
        super(AttnDecoder, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.gru = nn.GRU(hidden_size + output_size, hidden_size, n_layers, dropout=dropout_p, batch_first=True)
        self.audio_form = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, audio_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        bs = audio_input.size(0)

        # Combine  input audio and last context, run through RNN
        rnn_input = torch.cat((audio_input, last_context), dim=1)  # rnn_input = bs * (hidden*2)
        rnn_input = rnn_input.view(bs, 1, -1)  # rnn_input = bs * 1 * (hidden*2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        # rnn_output = bs * 1 * hidden, hidden = (num_layers * num_directions) * bs * hidden_size

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.view(bs, -1), encoder_outputs)
        attn_weights = attn_weights.view(bs, 1, -1)
        context = attn_weights.bmm(encoder_outputs)
        # (bs * 1 * seq) X (bs * seq * encoder_hidden_size) = bs * 1 * en_hidden_size
        context = context.view(bs, -1)  # context = bs * en_hidden_size

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.view(bs, -1)

        audio_output = self.audio_form(torch.cat((rnn_output, context), dim=1))
        audio_output = F.tanh(audio_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return audio_output, context, hidden, attn_weights


class pairwise_loss(nn.Module):
    def __init__(self):
        super(pairwise_loss, self).__init__()

    def forward(self, audio_output, target):
        dis = nn.functional.pairwise_distance(audio_output, target)
        loss = torch.mean(dis)
        return loss
