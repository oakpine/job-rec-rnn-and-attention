import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils.global_p import *

class RNN(RecModel):
    # TODO: not sure if these options are right
    append_id = True
    include_id = False
    include_user_features = False
    include_item_features = False

    @staticmethod
    def parse_model_args(parser, model_name='RNN'):
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in RNN.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of RNN layers.')
        parser.add_argument('--p_layers', type=str, default='[64]',
                            help="Size of each layer.")
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, hidden_size, num_layers, p_layers, *args, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_layers = p_layers
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        
        self.rnn = torch.nn.RNN(
            input_size=self.ui_vector_size, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layers)
        self.out = torch.nn.Linear(self.hidden_size, self.ui_vector_size, bias=False)

    def predict(self, feed_dict):
        check_list = []
        i_ids = feed_dict['X'][:, 1]
        history = feed_dict[C_HISTORY]

        his_vectors = self.iid_embeddings(history)

        # print(his_vectors.size())
        output, hidden = self.rnn(his_vectors, None)
        # print(output.size())
        # print(hidden.size())
        # print('hidden[-1]')
        # print(hidden[-1].size())
        output = self.out(hidden[-1])  # B * V
        # print(output.size())
        # print("\nnew")

        i_vectors = self.iid_embeddings(i_ids)

        prediction = (output * i_vectors).sum(dim=1).view([-1])
       
        check_list.append(('prediction', prediction))
        out_dict = {'prediction': prediction,
                    'check': check_list}
        # print(prediction)
        return out_dict