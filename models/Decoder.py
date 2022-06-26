from put_everthing_together import *

class Decoder(nn.Module):
    def __init__(self,n_layers, hidden_dim) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim) for _ in range(n_layers)])
    def forward(self,x1,x2,dec_att_mask,enc_att_mask):
        '''

        :param x1: decoder input: (N,L1,D)
        :param x2: encoder output: (N,L2,D)
        :param dec_att_mask: (N,L1)
        :param enc_att_mask: (N,L2)
        :return:   (N,L1,D)
        '''

        tmp = x1
        for f in self.layers:
            tmp = f.forward(tmp,x2,dec_att_mask,enc_att_mask)
        return tmp