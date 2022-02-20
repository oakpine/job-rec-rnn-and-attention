import torch
import torch.nn as nn 
import torch.nn.functional as F
from models.RecModel import RecModel
from utils.global_p import *
import math

class AbsPosEncoding(RecModel):
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='GRU'):
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors.')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, hidden_size, max_his, *args, **kwargs):
        self.hidden_size = hidden_size
        self.max_his = max_his
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        
        self.linear_1 = torch.nn.Linear(self.ui_vector_size, self.hidden_size)
        self.linear_2 = torch.nn.Linear(self.hidden_size, self.ui_vector_size)

        # positional encoding
        pe = torch.zeros(self.max_his, self.ui_vector_size)
        for pos in range(self.max_his):
            for i in range(0, self.ui_vector_size, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/self.ui_vector_size)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/self.ui_vector_size)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # for learned encodings
        #self.pos_embeddings = torch.nn.Embedding(self.max_his, self.ui_vector_size)
        
        heads = 3
        self.norm_1 = Norm(self.ui_vector_size)
        self.norm_2 = Norm(self.ui_vector_size)
        self.attn = MultiHeadAttention(heads, self.ui_vector_size)
        self.ff = FeedForward(self.ui_vector_size)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)

    def attention(self, query, key, value):
        query = self.q(query)
        key = self.q(key)
        value = self.q(value)

        scale = query.size(-1) ** 0.5 
        scores = query.bmm(key.transpose(1, 2)) / scale

        softmax = F.softmax(scores, dim=-1)
        return softmax.bmm(value)

    def predict(self, feed_dict):
        check_list = []
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[C_HISTORY]

        his_vectors = self.iid_embeddings(history)

        # positional embedding
        his_vectors = his_vectors * math.sqrt(self.ui_vector_size) # make embeddings relatively larger
        x = his_vectors + self.pe[:,:his_vectors.size(1)]
        #x = his_vectors + self.pos_embeddings(torch.arange(his_vectors.size(1))) # For learned encodings

        #his_vectors = his_vectors.view(his_vectors.size(0), -1)

        # add attention
        x = x + self.dropout_1(self.attn(x.clone(),x.clone(),x.clone()))
        x = self.norm_1(x)
        x = x + self.dropout_2(self.ff(x))
        output = self.norm_2(x)

        # get item embeddings
        i_vectors = self.iid_embeddings(i_ids)

        output = output[:, -1] # use last item embedding

        prediction = (output * i_vectors).sum(dim=1).view([-1])
       
        check_list.append(('prediction', prediction))
        out_dict = {'prediction': prediction,
                    'check': check_list}
        # print(prediction)
        return out_dict

class Norm(nn.Module):
  def __init__(self, dim_model, eps = 1e-6):
    super().__init__()

    self.size = dim_model
    # create two learnable parameters to calibrate normalisation
    self.alpha = nn.Parameter(torch.ones(self.size))
    self.bias = nn.Parameter(torch.zeros(self.size))
    self.eps = eps
  
  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
    / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm

class FeedForward(nn.Module):
  def __init__(self, dim_model: int = 215, dim_ff: int = 2048):
    super().__init__()
    self.linear_1 = nn.Linear(dim_model, dim_ff)
    self.dropout = nn.Dropout(0.1)
    self.linear_2 = nn.Linear(dim_ff, dim_model)

  def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, dim_model: int, dropout = 0.1):
        super().__init__()
        self.q = nn.Linear(dim_model, dim_model)
        self.k = nn.Linear(dim_model, dim_model)
        self.v = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query = self.q(query)
        key = self.q(key)
        value = self.q(value)

        scale = query.size(-1) ** 0.5
        scores = query.bmm(key.transpose(1, 2)) / scale

        softmax = F.softmax(scores, dim=-1)
        return softmax.bmm(value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_model: int, dropout = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_model) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_model, dim_model)

    def forward(self, query, key, value):
        return self.linear(torch.cat([h(query, key, value) for h in self.heads], dim=-1))