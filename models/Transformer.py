from models.layers import *
class Transformer(nn.Module):
    def __init__(self,n_layers,hidden_dim,n_labels) -> None:
        super().__init__()
        self.encoder = Encoder(n_layers,hidden_dim)
        self.decoder = Decoder(n_layers,hidden_dim)
        self.clf = ClassificationLayer(hidden_dim,n_labels)

    def forward(self,enc_input, dec_input,enc_att_mask,dec_att_mask):
        '''

        :param enc_input: (N,L_enc,D)
        :param dec_input: (N,L_dec,D)
        :param enc_att_mask: (N,L_enc)
        :param dec_att_mask: (N,L_dec)
        :return: (N,L_dec,n_labels)
        '''
        enc_output = self.encoder(enc_input,enc_att_mask)
        dec_output = self.decoder(dec_input,enc_output,dec_att_mask,enc_att_mask)
        probs = self.clf(dec_output)
        return probs