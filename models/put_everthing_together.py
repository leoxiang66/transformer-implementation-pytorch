import torch
import torch.nn as nn
import math




class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dim = hidden_dim

        # multi-head SA
        self.num_heads = 12
        self.dim_per_head = self.dim // self.num_heads  # 64
        self.Wq = nn.Linear(self.dim, self.num_heads*self.dim_per_head, bias=False)
        self.Wk = nn.Linear(self.dim, self.num_heads*self.dim_per_head, bias=False)
        self.Wv = nn.Linear(self.dim, self.num_heads*self.dim_per_head, bias=False)
        self.layerNorm_SA = nn.LayerNorm(self.dim)
        self.W_reshape_back = nn.Linear(self.num_heads*self.dim_per_head,self.dim)

        # FFN layer
        self.ffn1 = nn.Linear(self.dim,self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4,self.dim)
        self.act = nn.GELU()
        self.layerNorm_ffn = nn.LayerNorm(self.dim)

        # dropout
        self.att_drop_prob = 0.1
        self.state_drop_prob = 0.5
        self.att_drop = nn.Dropout(self.att_drop_prob)
        self.state_drop = nn.Dropout(self.state_drop_prob)

    def compute_att_mask(self,attention_mask):
        '''

        :param attention_mask:  (N,L)
        :return: (N,#heads, L,L)
        '''
        mask_logits = torch.zeros(attention_mask.size(0), self.num_heads, attention_mask.size(1), attention_mask.size(1))
        mask_logits = mask_logits + attention_mask[:, None, None, :]
        mask_logits = (1.0 - mask_logits) * -10000.
        return mask_logits


    def MultiHeadSelfAttention(self, x , attention_mask):
        '''

        :param x: (N,L,D)
        :param attention_mask: (N,L)
            1: normal token
            0: padding token
        :return: (N,L,D)
        '''
        '''
        Q,K,V:

        (N,L,(#heads * dph)) ->(N,#heads,L,dph)
        '''
        new_size = x.size()[:-1] + (self.num_heads,self.dim_per_head) # (N,L, #heads, dph)
        Q = self.Wq(x).view(*new_size).permute(0,2,1,3) # (N,#heads,L,dph)
        K = self.Wk(x).view(*new_size).permute(0,2,1,3)
        V = self.Wv(x).view(*new_size).permute(0,2,1,3)

        '''
        attention mask here:
        implementation idea: 一般来说, 在计算出attention_logits后, 在计算attention_score之前, 将那些要mask掉的padding entry的attention_logits减掉一个非常大的正数,
        这样它通过softmax之后的概率就很小了
        <=> 加上一个非常大的负数(叫做masked_logits)
        '''
        attention_logits = torch.matmul(Q,K.transpose(2,3))/math.sqrt(self.dim)
        attention_logits += self.compute_att_mask(attention_mask)
        attention_score = nn.Softmax(-1)(attention_logits)
        attention_score = self.att_drop(attention_score)
        O = torch.matmul(attention_score,V) # (N,#heads,L,dph)
        O = O.permute(0,2,1,3)  # (N,L, #heads, dph)
        O = O.contiguous().view(x.size(0),x.size(1),-1) # (N,L, #heads*dph)
        O = self.W_reshape_back(O) # (N,L,D)
        O = self.state_drop(O)
        O = self.layerNorm_SA(x + O)
        return O

    def FFN(self,x):
        tmp1 = self.act(self.ffn1(x))
        tmp2 = self.ffn2(tmp1)
        output = self.state_drop(tmp2)
        output = self.layerNorm_ffn(x+output)
        return output

    def forward(self, x,attention_mask):
        '''

        :param x: shape (N,L,D) N is batch size, L is the length of the sequence, D is the dimension of word embeddings
        :param attention_mask: (N,L)
        :return: shape (N,L,D)
        '''
        x = self.MultiHeadSelfAttention(x,attention_mask)
        x = self.FFN(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dim = hidden_dim

        # multi-head SA
        self.num_heads = 12
        self.dim_per_head = self.dim // self.num_heads
        self.Wq = nn.Linear(self.dim, self.num_heads * self.dim_per_head, bias=False)
        self.Wk = nn.Linear(self.dim, self.num_heads * self.dim_per_head, bias=False)
        self.Wv = nn.Linear(self.dim, self.num_heads * self.dim_per_head, bias=False)
        self.layerNorm_SA = nn.LayerNorm(self.dim)
        self.W_reshape_back_SA = nn.Linear(self.num_heads * self.dim_per_head, self.dim)

        # Multi-head CA
        self.Wq2 = nn.Linear(self.dim, self.num_heads * self.dim_per_head, bias=False)
        self.Wk2 = nn.Linear(self.dim, self.num_heads * self.dim_per_head, bias=False)
        self.Wv2 = nn.Linear(self.dim, self.num_heads * self.dim_per_head, bias=False)
        self.layerNorm_CA = nn.LayerNorm(self.dim)
        self.W_reshape_back_CA = nn.Linear(self.num_heads * self.dim_per_head, self.dim)


        # FFN
        self.ffn1 = nn.Linear(self.dim,self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4,self.dim)
        self.act = nn.GELU()
        self.layerNorm_ffn = nn.LayerNorm(self.dim)

        # dropout
        self.att_drop_prob = 0.1
        self.state_drop_prob = 0.5
        self.att_drop = nn.Dropout(self.att_drop_prob)
        self.state_drop = nn.Dropout(self.state_drop_prob)



    def _compute_SA_pad_mask(self, attention_mask):
        '''

        :param attention_mask:  (N,L1)
        :return: (N,#heads, L1,L1)
        '''
        mask = torch.zeros((attention_mask.size(0), self.num_heads, attention_mask.size(1),
                                  attention_mask.size(1)),dtype=torch.int32)
        mask = mask + attention_mask[:, None, None, :]
        return mask

    def _compute_att_subsequence_mask(self, x):
        '''

        :param x: (N,L1,D)
        :return: (N,#heads, L1,L1)
        '''
        mask = torch.zeros((x.size(0),self.num_heads,x.size(1),x.size(1)),dtype=torch.int32)
        ones = torch.tril(torch.ones((x.size(1), x.size(1)),dtype=torch.int32), diagonal=0)
        mask += ones
        return mask

    def _compute_SA_mask_logits(self, x, attention_mask):
        '''

        :param x: (N,L1,D)
        :param attention_mask: (N,L1)
        :return: (N,#heads, L1,L1)
        '''

        att_pad_mask = self._compute_SA_pad_mask(attention_mask)
        att_subseq_mask = self._compute_att_subsequence_mask(x)
        mask = att_pad_mask & att_subseq_mask
        mask_logits = (1.0 - mask) * -10000.0
        return mask_logits

    def _compute_CA_pad_mask(self,x,enc_att_mask):
        '''

        :param x: decoder input: (N,L1,D)
        :param enc_att_mask: (N,L2)
        :return: (N,#heads, L1,L2)
        '''
        mask = torch.zeros((x.size(0), self.num_heads, x.size(1),
                            enc_att_mask.size(1)), dtype=torch.int32)
        mask = mask + enc_att_mask[:, None, None, :]
        return mask

    def _compute_CA_mask_logits(self,x,enc_att_mask):
        '''

        :param x: decoder input: (N,L1,D)
        :param enc_att_mask: (N,L2)
        :return: (N,#heads, L1,L2)
        '''
        mask = self._compute_CA_pad_mask(x,enc_att_mask)
        mask_logits = (1.0 - mask) * -10000.0
        return mask_logits


    def MultiHeadSelfAttention(self, x,attention_mask):
        '''

        :param x: (N,L1,D)
        :return: (N,L1,D)
        '''
        '''
        Q,K,V:

        (N,L,(#heads * dph)) ->(N,#heads,L,dph)
        '''
        new_size = x.size()[:-1] + (self.num_heads, self.dim_per_head)  # (N,L1, #heads, dph)
        Q = self.Wq(x).view(*new_size).permute(0, 2, 1, 3)  # (N,#heads,L1,dph)
        K = self.Wk(x).view(*new_size).permute(0, 2, 1, 3)
        V = self.Wv(x).view(*new_size).permute(0, 2, 1, 3)

        attention_score = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.dim) # (N,#heads,L1, L1)
        attention_score += self._compute_SA_mask_logits(x,attention_mask)
        attention_score = nn.Softmax(-1)(attention_score)
        attention_score = self.att_drop(attention_score)
        O = torch.matmul(attention_score,V)
        O = O.permute(0, 2, 1, 3)  # (N,L1, #heads, dph)
        O = O.contiguous().view(x.size(0), x.size(1), -1)  # (N,L1, #heads*dph)
        O = self.W_reshape_back_SA(O)  # (N,L1,D)
        O = self.state_drop(O)
        O = self.layerNorm_SA(x + O)
        assert x.shape == O.shape
        return O


    def MultiHeadCrossAttention(self, x1, x2,enc_att_mask):
        '''

        :param x1: decoder input: (N,L1,D)
        :param x2: encoder output: (N,L2,D)
        :return: (N,L1,D)
        '''
        '''
        Q,K,V:

        (N,L,(#heads * dph)) ->(N,#heads,L,dph)
        '''
        N = x1.size(0)
        Q = self.Wq2(x1).view(N, -1, self.num_heads, self.dim_per_head).transpose(1, 2)  # Q: [N, n_heads, L1, dph]
        K = self.Wk2(x2).view(N, -1, self.num_heads, self.dim_per_head).transpose(1, 2)  # K: [N, n_heads, L2, dph]
        V = self.Wv2(x2).view(N, -1, self.num_heads, self.dim_per_head).transpose(1, 2)  # V: [N, n_heads, L2, dph]

        attention_score = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.dim)    #[N, n_heads, L1, L2]
        attention_score += self._compute_CA_mask_logits(x1,enc_att_mask)
        attention_score = nn.Softmax(-1)(attention_score)
        attention_score = self.att_drop(attention_score)
        O = torch.matmul(attention_score,V) # [N, n_heads, L1, dph]
        O = O.permute(0, 2, 1, 3)  # (N,L1, n_heads, dph)
        O = O.contiguous().view(N, x1.size(1), -1)  # (N,L1, n_heads*dph)
        O = self.W_reshape_back_CA(O)  # (N,L1,D)
        O = self.layerNorm_SA(x1 + O)
        assert O.shape == x1.shape
        return O


    def FFN(self,x):
        tmp1 = self.act(self.ffn1(x))
        tmp2 = self.ffn2(tmp1)
        tmp2 = self.state_drop(tmp2)
        output = self.layerNorm_ffn(x+tmp2)
        assert output.shape == x.shape
        return output

    def forward(self,x1,x2,dec_att_mask,enc_att_mask):
        '''

        :param x1: decoder input: (N,L1,D)
        :param x2: encoder output: (N,L2,D)
        :param dec_att_mask: (N,L1)
        :param enc_att_mask: (N,L2)
        :return:   (N,L1,D)
        '''

        x1 = self.MultiHeadSelfAttention(x1,dec_att_mask)
        tmp = self.MultiHeadCrossAttention(x1, x2,enc_att_mask)
        output = self.FFN(tmp)
        return output

class ClassificationLayer(nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln1 = nn.Linear(input_dim,input_dim*2)
        self.ln2 = nn.Linear(input_dim*2,output_dim)
        self.act = nn.ReLU()
        self.drop =nn.Dropout(0.2)

    def forward(self,x):
        '''

        :param x: (N,L,D)
        :return: prob. distribution (N,L,dout)
        '''
        x = self.ln1(x) # (N,L,D*2)
        x = self.act(x)
        logits = self.ln2(x) # (N,L,dout)
        logits = self.drop(logits)
        return logits


class Encoder(nn.Module):
    def __init__(self,n_layers,hidden_dim) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self,x,enc_att_mask):
        '''

        :param x: (N,L1,D)
        :return: (N,L1,D)
        '''
        tmp = x
        for f in self.layers:
            tmp = f.forward(tmp,enc_att_mask)

        return tmp

class Decoder(nn.Module):
    def __init__(self,n_layers, hidden_dim) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim) for _ in range(n_layers)])
    def forward(self,x1,x2,enc_att_mask,dec_att_mask):
        '''

        :param x1: decoder input: (N,L1,D)
        :param x2: encoder output: (N,L2,D)
        :param dec_att_mask: (N,L1)
        :param enc_att_mask: (N,L2)
        :return:   (N,L1,D)
        '''

        tmp = x1
        for f in self.layers:
            tmp = f.forward(tmp,x2,enc_att_mask,dec_att_mask)
        return tmp

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
        logits = self.clf(dec_output)
        return logits


class Loss(object):
    def __init__(self,weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0) -> None:
        super().__init__()
        self.CE = nn.CrossEntropyLoss(weight,size_average,ignore_index,reduce,reduction,label_smoothing)

    def __call__(self, dec_output, labels):
        '''

        :param dec_output: (N,L,n_labels): unnormalized logits
        :param labels: (N,L): class indices
        :return:
        '''

        dec_ouput = dec_output.view(-1,dec_output.size(-1)) # (N*L,n_labels)
        labels = labels.view(-1)    # (N*L)
        return self.CE(dec_ouput,labels)

